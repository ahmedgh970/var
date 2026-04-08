from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class VARTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: nn.Module | None,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        amp: bool = True,
        grad_clip: float = 1.0,
        save_dir: str = "checkpoints/var",
        log_file: str | None = None,
        is_main_process: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.amp = amp
        self.grad_clip = grad_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(log_file) if log_file is not None else None
        self.is_main_process = is_main_process
        self.scaler = GradScaler(enabled=self.amp)
        self.model.to(self.device)
        if self.tokenizer is not None:
            self.tokenizer.to(self.device)
            self.tokenizer.eval()

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _log(self, text: str):
        if not self.is_main_process:
            return
        print(text)
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

    def _sync(self, loss_sum: float, n: int):
        if not dist.is_available() or not dist.is_initialized():
            return loss_sum, n
        t = torch.tensor([loss_sum, float(n)], device=self.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0].item()), int(t[1].item())

    def _step(self, ms_tokens: list[torch.Tensor]):
        cond = None
        if self.tokenizer is not None:
            with torch.no_grad():
                cond = self.tokenizer.idx_to_var_input(ms_tokens)
        with autocast(enabled=self.amp):
            _, _, loss = self.model(ms_tokens, cond_blc_wo_first_l=cond)
        return loss

    def train_one_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        loss_sum = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"train {epoch}", leave=False, disable=not self.is_main_process)

        for ms_tokens in pbar:
            ms_tokens = [t.to(self.device, non_blocking=True) for t in ms_tokens]
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step(ms_tokens)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = ms_tokens[0].shape[0]
            loss_sum += float(loss.detach()) * bs
            n += bs
            pbar.set_postfix(loss=f"{loss_sum / max(1, n):.4f}")

        loss_sum, n = self._sync(loss_sum, n)
        return {"loss": loss_sum / max(1, n)}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        loss_sum = 0.0
        n = 0
        pbar = tqdm(val_loader, desc=f"val {epoch}", leave=False, disable=not self.is_main_process)

        for ms_tokens in pbar:
            ms_tokens = [t.to(self.device, non_blocking=True) for t in ms_tokens]
            loss = self._step(ms_tokens)
            bs = ms_tokens[0].shape[0]
            loss_sum += float(loss.detach()) * bs
            n += bs

        loss_sum, n = self._sync(loss_sum, n)
        return {"loss": loss_sum / max(1, n)}

    def save_checkpoint(self, epoch: int, best: bool = False):
        if not self.is_main_process:
            return
        ckpt = {
            "epoch": epoch,
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(ckpt, self.save_dir / "last.pt")
        if best:
            torch.save(ckpt, self.save_dir / "best.pt")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        eval_every: int = 1,
        save_every: int = 1,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
    ):
        best_val = float("inf")
        best_epoch = -1
        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(train_loader, epoch)

            val_stats = None
            if eval_every > 0 and epoch % eval_every == 0:
                val_stats = self.evaluate(val_loader, epoch)
                if val_stats["loss"] < (best_val - early_stopping_min_delta):
                    best_val = val_stats["loss"]
                    best_epoch = epoch
                    no_improve_count = 0
                    self.save_checkpoint(epoch, best=True)
                    self._log(f"[epoch {epoch}] new best: val_loss={val_stats['loss']:.4f} -> saved best.pt")
                else:
                    no_improve_count += 1
                    self._log(
                        f"[epoch {epoch}] no best update: val_loss={val_stats['loss']:.4f}, best={best_val:.4f} (epoch {best_epoch})"
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            if save_every > 0 and epoch % save_every == 0:
                self.save_checkpoint(epoch, best=False)
                self._log(f"[epoch {epoch}] saved last.pt")

            if val_stats is None:
                self._log(f"[epoch {epoch}] train loss={train_stats['loss']:.4f}")
            else:
                self._log(f"[epoch {epoch}] train loss={train_stats['loss']:.4f} | val loss={val_stats['loss']:.4f}")

            if early_stopping_patience > 0 and val_stats is not None and no_improve_count >= early_stopping_patience:
                self._log(
                    f"[epoch {epoch}] early stopping triggered: no improvement for {no_improve_count} evals "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
                )
                break
