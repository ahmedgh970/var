import os

import torch
import torch.distributed as dist


def init_distributed(device_type: str) -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_ddp = world_size > 1
    if use_ddp:
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if device_type == "cuda":
            torch.cuda.set_device(local_rank)
    return use_ddp, rank, local_rank
