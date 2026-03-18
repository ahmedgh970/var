from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import tokenizer_total_loss


class TokenizerTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        amp: bool = True,
        grad_clip: float = 1.0,
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        loss_type: str = "mse",
        save_dir: str = "checkpoints/tokenizer",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.amp = amp
        self.grad_clip = grad_clip
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.loss_type = loss_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler(enabled=self.amp)

        self.model.to(self.device)

    def _step(self, images: torch.Tensor):
        with autocast(enabled=self.amp):
            recon, _, vq_loss = self.model(images)
            total, rec_loss, vq_loss = tokenizer_total_loss(
                recon=recon,
                target=images,
                vq_loss=vq_loss,
                recon_weight=self.recon_weight,
                vq_weight=self.vq_weight,
                loss_type=self.loss_type,
            )
        return total, rec_loss, vq_loss

    def train_one_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        total_meter = 0.0
        rec_meter = 0.0
        vq_meter = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}", leave=False)
        for batch in pbar:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            total, rec_loss, vq_loss = self._step(images)
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = images.shape[0]
            total_meter += float(total.detach()) * bs
            rec_meter += float(rec_loss.detach()) * bs
            vq_meter += float(vq_loss.detach()) * bs
            n += bs
            pbar.set_postfix(
                total=f"{total_meter / max(1, n):.4f}",
                rec=f"{rec_meter / max(1, n):.4f}",
                vq=f"{vq_meter / max(1, n):.4f}",
            )

        return {
            "total": total_meter / max(1, n),
            "recon": rec_meter / max(1, n),
            "vq": vq_meter / max(1, n),
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        total_meter = 0.0
        rec_meter = 0.0
        vq_meter = 0.0
        n = 0

        pbar = tqdm(val_loader, desc=f"val {epoch}", leave=False)
        for batch in pbar:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(self.device, non_blocking=True)
            total, rec_loss, vq_loss = self._step(images)

            bs = images.shape[0]
            total_meter += float(total.detach()) * bs
            rec_meter += float(rec_loss.detach()) * bs
            vq_meter += float(vq_loss.detach()) * bs
            n += bs

        return {
            "total": total_meter / max(1, n),
            "recon": rec_meter / max(1, n),
            "vq": vq_meter / max(1, n),
        }

    def save_checkpoint(self, epoch: int, best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        last_path = self.save_dir / "last.pt"
        torch.save(ckpt, last_path)
        if best:
            torch.save(ckpt, self.save_dir / "best.pt")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        eval_every: int = 1,
        save_every: int = 1,
    ):
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_stats = self.train_one_epoch(train_loader, epoch)

            val_stats = None
            if eval_every > 0 and epoch % eval_every == 0:
                val_stats = self.evaluate(val_loader, epoch)
                if val_stats["total"] < best_val:
                    best_val = val_stats["total"]
                    self.save_checkpoint(epoch, best=True)

            if self.scheduler is not None:
                self.scheduler.step()

            if save_every > 0 and epoch % save_every == 0:
                self.save_checkpoint(epoch, best=False)

            if val_stats is None:
                print(
                    f"[epoch {epoch}] "
                    f"train total={train_stats['total']:.4f} "
                    f"recon={train_stats['recon']:.4f} "
                    f"vq={train_stats['vq']:.4f}"
                )
            else:
                print(
                    f"[epoch {epoch}] "
                    f"train total={train_stats['total']:.4f} "
                    f"recon={train_stats['recon']:.4f} "
                    f"vq={train_stats['vq']:.4f} | "
                    f"val total={val_stats['total']:.4f} "
                    f"recon={val_stats['recon']:.4f} "
                    f"vq={val_stats['vq']:.4f}"
                )
