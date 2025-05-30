import torch


def get_optimizer(model: torch.nn.Module, lr: float, wd: float,
                  optimizer_name: str = 'adamw'):
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError(f"Unkown optimizer: {optimizer_name}")
    return optimizer


def get_lr_scheduler(optimizer: torch.optim.Optimizer, max_steps: int, max_lr: float, min_lr: float,
                     warmup_steps: int, scheduler_name: str = 'cosine'):
    if scheduler_name == "cosine":
        if warmup_steps is None:
            # Only cosine decay without warm-up
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_lr)
            print(f"!! Using cosine learning rate decay...")
        else:
            # Linear warm-up phase parameters
            end_factor = 1.0
            start_factor = end_factor / warmup_steps  # linear step size
            linear_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor,
                                                                    total_iters=warmup_steps)

            # Remaining steps after linear warm-up
            remaining_steps = max_steps - warmup_steps
            cosine_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=min_lr)

            # Sequentially combine warm-up and cosine schedulers
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[linear_lr_scheduler, cosine_lr_scheduler],
                milestones=[warmup_steps],
            )
            print(f"!! Using cosine learning rate decay with a linear warmup for {warmup_steps} steps...")
    elif scheduler_name == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=min_lr/max_lr, total_iters=max_steps)
        print(f"!! Using linear learning rate decay...")
    else:
        raise NotImplementedError(f"Unknown LR scheduler: {scheduler_name}")

    return lr_scheduler


def get_num_model_params(model):
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters() if p.requires_grad]
    )
