# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from .dataset import FSC147

def build_loader(config, mode):
    data_path = config.DATA_PATH
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    train_set = FSC147(data_path, mode)

    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler = DistributedSampler(
    #         train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )

    return DataLoader(
        train_set,
        batch_size = batch_size,
        #sampler=sampler,
        num_workers = num_workers,
        pin_memory=config.PIN_MEMORY,
        shuffle = (mode=='train'),
        # drop_last = mode=='Train'
        collate_fn=FSC147.collate_fn
    )

def build_normal_loader(config, mode):
    data_path = config.DATA_PATH
    batch_size = config.BATCH_SIZE
    train_set = FSC147(data_path, mode)

    return DataLoader(
        train_set,
        batch_size = batch_size,
        num_workers = 4,
        pin_memory=config.PIN_MEMORY,
        shuffle = False,
        #drop_last = mode=='Train'
        collate_fn=FSC147.collate_fn
    )
