import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from longvu.mm_datautils import get_mm_adapter_state_maybe_zero_3
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length
from peft import PeftModelForCausalLM
import deepspeed
from collections import defaultdict

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


def get_mm_adapter_state_maybe_zero_3(
    named_params: Dict[str, torch.Tensor], keys_to_match: List[str]
) -> Dict[str, torch.Tensor]:
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()  # pyre-ignore
        for k, v in to_return.items()
    }
    return to_return


def maybe_zero_3(
    param: torch.Tensor, ignore_status: bool = False, name: Optional[str] = None
) -> torch.Tensor:
    return param.detach().cpu().clone()

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

class RRPOTrainer(Trainer):
    """
    a base class for preference optimization training
    
    """
    def __init__(
        self,
        model,
        ref_model,
        args,
        data_collator,
        loss_fn,
        train_dataset,
        tokenizer,
        disable_dropout: bool = True,
        callbacks = [],
        train_dataloader = None,
        **kwargs
    ):
        self.ref_model = ref_model
        
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self._stored_metrics = {}
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataloader is not None:
            print("Using sonic dataloader")
            return self.accelerator.prepare(self.train_dataloader)
        # pyre-fixme[16]: `Trainer` has no attribute `get_train_dataloader`.
        return super().get_train_dataloader()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # pyre-fixme[16]: `Trainer` has no attribute `_get_train_sampler`.
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        # pyre-fixme[16]: `Trainer` has no attribute `optimizer`.
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # pyre-fixme[16]: `Trainer` has no attribute `mm_projector_lr`.
            assert not (self.args.mm_projector_lr and self.args.mm_vision_sampler_lr)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.mm_vision_sampler_lr is not None:
                vision_sampler_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if ("vision_sampler" in name) or ("vision_query" in name)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                ]
            elif (
                self.args.unfreeze_mm_vision_tower
                and self.args.mm_vision_tower_lr is not None
            ):
                vision_tower_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "vision_tower" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
        return self.optimizer

    # pyre-fixme[2]: Parameter must be annotated.
    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        if getattr(self.args, "tune_mm_mlp_adapter", False):

            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(),
                keys_to_match,
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        else:
            super(RRPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(RRPOTrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, 
        model=None,
        inputs=None,
        return_outputs=False,
    ):

        align_dict = self.concatenated_forward(self.model, inputs)
        with torch.no_grad():
            ref_dict = self.concatenated_forward(self.ref_model, inputs)

        # # calculate loss
        loss, metrices = self.loss_fn(align_dict, ref_dict, self.args.local_rank, self.args.world_size)

        MB = 1024.0 * 1024.0
        GB = 1024.0 * 1024.0 * 1024.0
        print_string=""
        self._stored_metrics['train']=defaultdict(list)
        for key in metrices:
            value = round(sum(metrices[key])/len(metrices[key]), 4)
            print_string+=f"{key}: {value} "
            self._stored_metrics['train'][key].append(metrices[key])

        print_string+=f"max mem: {torch.cuda.max_memory_allocated() // GB}"
        print(print_string)

        return loss
    
    def concatenated_forward(self, 
        model, inputs
    ):

        images = inputs["images"]
        image_sizes = inputs['image_sizes']
        image_aux_attention_masks_list = inputs['image_aux_attention_masks_list']

        neg_images = inputs["neg_images"]
        neg_image_sizes = inputs['neg_image_sizes']
        neg_image_aux_attention_masks_list = inputs['neg_image_aux_attention_masks_list']

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]

        neg_input_ids = inputs["neg_input_ids"]
        neg_labels = inputs["neg_labels"]
        neg_attention_mask = inputs["neg_attention_mask"]
        neg_position_ids = inputs["neg_position_ids"]

        signs = inputs["signs"]
        neg_signs = inputs["neg_signs"]
            
        max_dim = max(input_ids.shape[1], neg_input_ids.shape[1])
        batch_input_ids = torch.zeros((input_ids.shape[0]*2, max_dim), dtype=input_ids.dtype, device=input_ids.device)
        batch_labels = torch.ones((input_ids.shape[0]*2, max_dim), dtype=labels.dtype, device=labels.device) * -100
        batch_attention_mask = torch.zeros((input_ids.shape[0]*2, max_dim), device=attention_mask.device).to(torch.bool)
        batch_signs = torch.zeros((input_ids.shape[0]*2, max_dim), dtype=signs.dtype, device=signs.device)
        batch_position_ids = torch.zeros((input_ids.shape[0]*2, max_dim), device=position_ids.device, dtype=position_ids.dtype)
        
        batch_input_ids[:input_ids.shape[0], :input_ids.shape[1]] = input_ids
        batch_input_ids[neg_input_ids.shape[0]:, :neg_input_ids.shape[1]] = neg_input_ids
        batch_labels[:labels.shape[0], :labels.shape[1]] = labels
        batch_labels[neg_labels.shape[0]:, :neg_labels.shape[1]] = neg_labels
        batch_attention_mask[:attention_mask.shape[0], :attention_mask.shape[1]] = attention_mask
        batch_attention_mask[neg_attention_mask.shape[0]:, :neg_attention_mask.shape[1]] = neg_attention_mask
        batch_signs[:signs.shape[0], :signs.shape[1]] = signs
        batch_signs[neg_signs.shape[0]:, :neg_signs.shape[1]] = neg_signs
        batch_position_ids[:position_ids.shape[0], :position_ids.shape[1]] = position_ids
        batch_position_ids[neg_position_ids.shape[0]:, :neg_position_ids.shape[1]] = neg_position_ids
        # two types of images dino and siglip
        batch_images=[torch.cat([images[i], neg_images[i]], dim=0).to(model.dtype) for i in range(len(images))]
        batch_image_aux_attention_masks_list=[torch.cat([image_aux_attention_masks_list[i], neg_image_aux_attention_masks_list[i]], dim=0) for i in range(len(image_aux_attention_masks_list))],
        batch_image_sizes=image_sizes+neg_image_sizes
        
        (
            all_logits,
            batch_labels,
            batch_signs,
        ) = model( 
            input_ids=batch_input_ids,
            position_ids=batch_position_ids,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=batch_images,
            image_aux_attention_masks_list=batch_image_aux_attention_masks_list,
            image_sizes=batch_image_sizes,
            signs=batch_signs,
            forward_mode='rrpo',
        )

        return dict(logits=all_logits, 
                    targets=batch_labels, 
                    signs=batch_signs)
        
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"

        if train_eval not in self._stored_metrics: # we don't have eval
            return super().log(logs)

        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            # logs[key] = torch.tensor(metrics).mean().item()
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)