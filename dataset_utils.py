#!/bin/python

import random

import torch
import numpy as np


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int = 8,
                   drop_last: bool = False, pin_loader_memory: bool = True,
                   persistent_workers: bool = True, prefetch_factor: int = 2,
                   generator: torch.Generator = None, shuffle: bool = True):
    sampler = None
    if torch.distributed.is_initialized():
        print("!! Attaching sampler to the DataLoader for distributed training...")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # shuffling is done by the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             sampler=sampler, drop_last=drop_last, pin_memory=pin_loader_memory,
                                             worker_init_fn=seed_worker, persistent_workers=persistent_workers,
                                             prefetch_factor=prefetch_factor, generator=generator)
    return dataloader
