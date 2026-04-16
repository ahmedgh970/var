import torch
from torch.utils.data.sampler import Sampler


class InfiniteBatchSampler(Sampler):
    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        seed_for_all_rank: int = 0,
        fill_last: bool = False,
        shuffle: bool = True,
        start_ep: int = 0,
        start_it: int = 0,
    ):
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.iters_per_ep = (self.dataset_len + self.batch_size - 1) // self.batch_size
        self.max_p = self.iters_per_ep * self.batch_size
        self.fill_last = bool(fill_last)
        self.shuffle = bool(shuffle)
        self.epoch = int(start_ep)
        self.same_seed_for_all_ranks = int(seed_for_all_rank)
        self.indices = self._generate_indices()
        self.start_ep = int(start_ep)
        self.start_it = int(start_it)

    def _generate_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            indices = torch.randperm(self.dataset_len, generator=g)
        else:
            indices = torch.arange(self.dataset_len)

        if self.fill_last and indices.numel() < self.max_p:
            fill = self.max_p - indices.numel()
            indices = torch.cat((indices, indices[:fill]), dim=0)

        return indices.tolist()

    def __iter__(self):
        self.epoch = self.start_ep
        while True:
            self.epoch += 1
            p = (self.start_it * self.batch_size) if self.epoch == self.start_ep else 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self._generate_indices()

    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(
        self,
        world_size: int,
        rank: int,
        dataset_len: int,
        glb_batch_size: int,
        same_seed_for_all_ranks: int = 0,
        fill_last: bool = False,
        shuffle: bool = True,
        start_ep: int = 0,
        start_it: int = 0,
    ):
        if glb_batch_size % world_size != 0:
            raise ValueError("glb_batch_size must be divisible by world_size")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.dataset_len = int(dataset_len)
        self.glb_batch_size = int(glb_batch_size)
        self.batch_size = self.glb_batch_size // self.world_size

        self.iters_per_ep = (self.dataset_len + self.glb_batch_size - 1) // self.glb_batch_size
        self.fill_last = bool(fill_last)
        self.shuffle = bool(shuffle)
        self.epoch = int(start_ep)
        self.same_seed_for_all_ranks = int(same_seed_for_all_ranks)
        self.indices = self._generate_indices()
        self.start_ep = int(start_ep)
        self.start_it = int(start_it)

    def _generate_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            global_indices = torch.randperm(self.dataset_len, generator=g)
        else:
            global_indices = torch.arange(self.dataset_len)

        if self.fill_last and global_indices.numel() < global_max_p:
            fill = global_max_p - global_indices.numel()
            global_indices = torch.cat((global_indices, global_indices[:fill]), dim=0)

        seps = torch.linspace(0, global_indices.shape[0], self.world_size + 1, dtype=torch.int64)
        beg = seps[self.rank].item()
        end = seps[self.rank + 1].item()
        local_indices = global_indices[beg:end].tolist()
        self.max_p = len(local_indices)
        return local_indices
