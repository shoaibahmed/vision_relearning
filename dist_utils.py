import os
import pickle
from argparse import Namespace
from datetime import timedelta
from collections import OrderedDict

import torch
import torch.distributed.fsdp


def init_distributed_env(args: Namespace):
    # Initialize the distributed environment
    args.world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))
    args.distributed = args.world_size > 1
    args.rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    args.local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    args.gpu = args.local_rank

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=1))
        obtained_world_size = torch.distributed.get_world_size()
        assert obtained_world_size == args.world_size, f"{obtained_world_size} != {args.world_size}"
        print(f"Initializing the environment with {args.world_size} processes / Process rank: {args.rank} / Local rank: {args.local_rank}")
        setup_for_distributed(args.local_rank == 0)  # print via one process per node
    if not hasattr(args, 'effective_batch_size') or args.effective_batch_size is None:
        args.effective_batch_size = args.batch_size * args.world_size
        print("Setting effective batch size to be:", args.effective_batch_size)
    print(f"# processes: {args.world_size} / batch size: {args.batch_size} / effective batch size: {args.effective_batch_size}")


def is_main_proc(local_rank=None, shared_fs=True):
    assert shared_fs or local_rank is not None
    main_proc = not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0 if shared_fs else local_rank == 0)
    return main_proc


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def wait_for_other_procs():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def convert_to_distributed(model, local_rank, use_ddp=True, sync_bn=False, broadcast_buffers=True,
                           find_unused_parameters=False, fsdp_auto_wrap_policy=None,
                           sharding_strategy=torch.distributed.fsdp.ShardingStrategy.HYBRID_SHARD):
    if not broadcast_buffers:  # important when doing multi-step autoregressive finetuning
        print("Turning on sync BN as DDP broadcast buffers is false")
        sync_bn = True  # sync BN takes care of synchronization
    if torch.distributed.is_initialized():
        if sync_bn:
            print("!! Using synced BN!")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if use_ddp:
            print(f"!! Wrapping model into DDP / find unused params: {find_unused_parameters} / broadcast buffers: {broadcast_buffers}")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                              broadcast_buffers=broadcast_buffers,
                                                              find_unused_parameters=find_unused_parameters)
        else:
            print("!! Wrapping model into FSDP...")
            print("FSDP sharding strategy:", sharding_strategy)
            model = torch.distributed.fsdp.FullyShardedDataParallel(model, device_id=local_rank, cpu_offload=None, mixed_precision=None,
                                                                    sharding_strategy=sharding_strategy, auto_wrap_policy=fsdp_auto_wrap_policy)
    return model


def reduce_tensor(tensor, average=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= world_size
    return rt


def gather_tensor(data):
    """
    Imported from: https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/util/misc.py#L88
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    if torch.is_tensor(data):
        main_device = data.device
    else:
        main_device = torch.device("cuda")
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
        if torch.is_tensor(data_list[-1]):
            data_list[-1] = data_list[-1].to(main_device)
    return data_list


def convert_state_dict(state_dict: OrderedDict, require_module: bool = None) -> OrderedDict:
    # Create new OrderedDict from the checkpoint state that does or does not contain "module." based on the model state
    if require_module is None:
        require_module = torch.distributed.is_initialized()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if require_module and not name.startswith("module."):
            name = "module." + k  # add module.
        elif not require_module and name.startswith("module."):
            name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict
