""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def create_scheduler(args, optimizer):
    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_training_steps,
            num_cycles=0.5,
            min_lr_multi=args.min_lr_multi
        )
    elif args.sched == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_training_steps,
            min_lr_multi=args.min_lr_multi
        )
    return lr_scheduler


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, min_lr_multi: float = 0., last_epoch: int = -1
):
    """
    Modified from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        min_lr_multi (`float`, *optional*, defaults to 0):
            The minimum learning rate multiplier. Thus the minimum learning rate is base_lr * min_lr_multi.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr_multi, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_multi, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_constant_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        # num_cycles: float = 0.5, 
        min_lr_multi: float = 0., last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr_multi, float(current_step) / float(max(1, num_warmup_steps)))
        return 1.0
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# FIXME: in the
# the steps are treated as epoch
# import matplotlib.pyplot as plt
# scheduler=LambdaLR(optimizer, lr_lambda, last_epoch)

# values=[]
# for k in range(1800):
#     optimizer.step()
#     scheduler.step()
#     values.append(optimizer.param_groups[0]["lr"])
# plt.plot(values)
# plt.savefig('lrs.png')

def warmup_cosine_scheduler(base_value, final_value, 
                     total_iters, warmup_iters,
                     start_warmup_value=0):

    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule
