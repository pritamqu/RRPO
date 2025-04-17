import datetime
import logging
import time
from os.path import join
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import torch.nn.functional as F
import sys
from peft import get_peft_model, LoraConfig, TaskType
sys.path.append('./videochat2/')
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from data import RRPODataset, setup_transformation
from utils.config import Config
from models.load_mistral import get_sinusoid_encoding_table
from models.videochat_mistra.videochat2_rrpo_mistral import VideoChat2_RRPO_Mistral
from losses import *
logger = logging.getLogger(__name__)


def load_pretrained_model(model_path, num_frames=16):

    config_file_mistral = "./videochat2/configs/config_mistral.json"
    cfg=Config.from_file(config_file_mistral)
    resolution = 224

    # load stage2 model
    cfg.model.vision_encoder.num_frames = 4
    model = VideoChat2_RRPO_Mistral(config=cfg.model)

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )
    model.mistral_model = get_peft_model(model.mistral_model, peft_config)
    state_dict = torch.load(f"{model_path}/videochat2_mistral_7b_stage3.pth", "cpu")

    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    
    print(msg)

    # try merging here
    logger.info('Merging LoRA weights...')
    model.mistral_model=model.mistral_model.merge_and_unload()

    # update pos_embed size 
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frames, cur_frame=num_frames)
    model.vision_encoder.encoder.pos_embed = new_pos_emb
    model = model.eval()
    
    # training specific setup
    print('use_cache: ', model.mistral_model.config.use_cache)
    model.mistral_model.config.use_cache = False
    print('setting use_cache: ', model.mistral_model.config.use_cache)

    return model, resolution, num_frames, cfg

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def train(
    model,
    ref_model,
    train_loader,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    loss_fn,
):
    
    model.train()
    ref_model.eval()

    # print('*'*10, 'print trainable params', '*'*10)
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)

    gradient_accumulation_steps = config.gradient_accumulation_steps


    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    # loss_names = ["loss"]

    # for name in loss_names:
    #     metric_logger.add_meter(
    #         f"{name}", SmoothedValue(window=1, fmt="{value:.4f}")
    #     )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        train_loader.sampler.set_epoch(epoch)
        
    iterator = metric_logger.log_every_multi_loader(train_loader, log_freq, header, len(train_loader))
    for i, batch in enumerate(iterator):
        video=batch[0].to(device, non_blocking=True)
        text_input={}
        text_input['pos_conversation']=batch[1] 
        text_input['neg_conversation']=batch[2]
        instruction=batch[3]
                
        with torch.cuda.amp.autocast(enabled=config.fp16):
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            align_dict = model(video, text_input, instruction, mode='rrpo')
            ref_dict = ref_model(video, text_input, instruction, mode='rrpo')
            
        loss, metrics = loss_fn(align_dict=align_dict, 
                        ref_dict=ref_dict, 
                        rank=config.rank, 
                        world_size=config.world_size,
                        )
                    
        loss = loss / gradient_accumulation_steps
        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        if (i + 1) % gradient_accumulation_steps == 0:
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()


        # logging
        for name in metrics:
            if name not in metric_logger.meters:
                metric_logger.add_meter(
                    f"{name}", SmoothedValue(window=1, fmt="{value:.4f}")
                    )
            value = round(sum(metrics[name])/len(metrics[name]), 4)
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{name}": value})

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if config.debug and global_step % 10 == 0:
            logger.info("debug mode, break training loop")
            break


    # gather the stats from all processes
    metric_logger.synchronize_between_processes() # i may skip it here as already there is a sync inside losses
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step


def create_dataloader(dataset, 
                      batch_size,
                      num_workers=4,
                      distributed=False, 
                      use_shuffle=True, 
                      collate_fn=None):

    
    if distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=use_shuffle
        )
        shuffle=False
    else:
        sampler=None
        shuffle=use_shuffle

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )
    
    return loader


def setup_data(config, tokenizer, resolution):

    train_dataset = RRPODataset(ann_file=[config.data_path, 
                                                 config.media_root, 
                                                 "video"], 
                                       transform=setup_transformation(resolution=resolution), 
                                       num_frames=config.num_frames, 
                                       tokenizer=tokenizer)
    
    logging.info(f"number of training samples {len(train_dataset)}")

    train_loader = create_dataloader(train_dataset, 
                      batch_size=config.batch_size,
                      num_workers=config.num_workers,
                      distributed=config.distributed, 
                      use_shuffle=True, 
                      collate_fn=None)

    return train_loader

def setup_model(
    config, model, find_unused_parameters=False
):

    model = model.to(torch.device(config.device))
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters,  # `False` for image-only task
        )

    start_epoch = 0
    global_step = 0

    optimizer = create_optimizer(config.optimizer, model)
    scheduler = create_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    )

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = []
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def setup_trainable_params(model, config):

    # selectively turn off gradients
    logger.info('Turning off gradients expect for the following')
    trainable_modules=()
    trainable_modules2=()
    if config.trainable_modules:
        trainable_modules=tuple(config.trainable_modules)
        trainable_modules2=tuple(['module.'+m for m in config.trainable_modules])

    for n, p in model.named_parameters():
        if n.startswith(trainable_modules) or n.startswith(trainable_modules2):
            p.requires_grad=True
            p=p.to(torch.float32)
            logger.info(n)
        else:
            p.requires_grad=False
        
    logger.info("Adding lora")
    
    # setup from videochat2
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    #     r=config.model.lora_r, lora_alpha=config.model.lora_alpha, lora_dropout=config.model.lora_dropout,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    #                         "gate_proj", "up_proj", "down_proj", "lm_head"]
    # )

    peft_config = LoraConfig(
        r=config.model.lora_r, 
        lora_alpha=config.model.lora_alpha, 
        target_modules=find_all_linear_names(model.mistral_model),
        lora_dropout=config.model.lora_dropout,
        bias=config.model.lora_bias,
        task_type="CAUSAL_LM",
    )

    model.mistral_model = get_peft_model(model.mistral_model, peft_config)

    print('current lora weights')
    print(list(model.mistral_model.named_parameters())[2])

    for n, p in model.mistral_model.named_parameters():
        if p.requires_grad:
            print(n, p.dtype)

    model.use_lora=True
    peft_config.save_pretrained(config.output_dir)

    return model

def save_model(model_without_ddp, config):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
    }
    state_dict = model_without_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    torch.save(state_dict, join(config.output_dir, "model.pth"))

def save_ckpt(model_without_ddp, optimizer, scheduler, scaler, config, epoch, global_step):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
    }
    state_dict = model_without_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
    }
    if config.get("save_latest", False):
        torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
    else:
        torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    # setup models
    logger.info('Loading online model')
    model, resolution, num_frame, model_cfg = load_pretrained_model(config.model_name_or_path, 
                                                                            num_frames=config.num_frames)
    
    model = setup_trainable_params(model, config)

    logger.info(f'Loading reference model: {config.model_name_or_path}')
    ref_model = load_pretrained_model(config.model_name_or_path, 
                                              num_frames=config.num_frames)[0]

    ref_model=ref_model.to(torch.device(config.device))
    # freeze reference model
    for n,p in ref_model.named_parameters():
        p.requires_grad = False

    disable_dropout_in_model(model)
    disable_dropout_in_model(ref_model)

    train_loader = setup_data(config, 
                            model.mistral_tokenizer, 
                            resolution)

    num_steps_per_epoch = len(train_loader)//config.gradient_accumulation_steps
    config.scheduler.num_training_steps = int(num_steps_per_epoch * config.scheduler.epochs)
    config.scheduler.num_warmup_steps = int(num_steps_per_epoch * config.scheduler.warmup_epochs)
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = True

    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model=model,
        find_unused_parameters=True,
    )

    # if is_main_process() and config.wandb.enable:
    #     wandb.watch(model)

    logger.info("Start training")
    start_time = time.time()

    loss_fn = RRPO(alpha=config.loss_alpha, 
                        beta=config.loss_beta, 
                        )
    
    for epoch in range(start_epoch, config.scheduler.epochs):

        global_step = train(
            model,
            ref_model,
            train_loader,
            optimizer,
            epoch,
            global_step,
            device,
            scheduler,
            scaler,
            config,
            loss_fn,
        )

        if is_main_process():
            logger.info(f"Epoch {epoch}")
            # just saving the model weights
            save_model(model_without_ddp, config)
            # save_ckpt(model_without_ddp, optimizer, scheduler, scaler, config, epoch, global_step)

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
