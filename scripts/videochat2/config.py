# ========================= data ==========================
data_path=""
media_root=""
num_frames=16
num_workers = 6
batch_size = 4
gradient_accumulation_steps=4
trainable_modules=[] # pass modules other than lora
# trainable_modules=['mistral_proj'] # pass modules other than lora
# ========================= training ==========================
model_name_or_path="/h/pritam/pritam_ssd004/.cache/huggingface/hub/VideoChat2_stage3_Mistral_7B"
model = dict(
    lora_r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    lora_bias="none",
)
loss_alpha=0.1
loss_beta=0.5
optimizer = dict(
    opt="adamW",
    lr=2e-5,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.05)
fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    entity="pritamqu",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="RRPO",  # setup in your command line
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?
