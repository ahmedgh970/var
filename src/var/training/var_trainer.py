from pathlib import Path
import math
from typing import Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from var.training.ema import ModelEMA


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
        grad_accum_steps: int = 1,
        weight_decay_start: float = 0.05,
        weight_decay_end: float = 0.05,
        schedule_name: str = "none",
        warmup_epochs: int = 0,
        final_lr_ratio: float = 0.1,
        lr_warmup_start_ratio: float = 0.005,
        progressive_ratio: float = 0.0,
        progressive_start_stage: int = 0,
        progressive_warmup_epochs: float = 0.0,
        save_dir: str = "checkpoints/var",
        log_file: str | None = None,
        is_main_process: bool = True,
        ema: ModelEMA | None = None,
        sample_fn: Callable[[nn.Module, Path, int], None] | None = None,
        num_val_samples: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.amp = amp
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.weight_decay_start = float(weight_decay_start)
        self.weight_decay_end = float(weight_decay_end)
        self.schedule_name = str(schedule_name).lower()
        self.warmup_epochs = int(warmup_epochs)
        self.final_lr_ratio = float(final_lr_ratio)
        self.lr_warmup_start_ratio = float(lr_warmup_start_ratio)
        self.progressive_ratio = float(progressive_ratio)
        self.progressive_start_stage = int(progressive_start_stage)
        self.progressive_warmup_epochs = float(progressive_warmup_epochs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(log_file) if log_file is not None else None
        self.is_main_process = is_main_process
        self.ema = ema
        self.sample_fn = sample_fn
        self.num_val_samples = num_val_samples
        self.scaler = GradScaler(enabled=self.amp)
        self.base_lrs = [float(pg["lr"]) for pg in self.optimizer.param_groups]
        self._warmup_steps = 0
        core_model = self._unwrap_model()
        self.begin_ends = list(core_model.begin_ends)
        self.seq_len = int(core_model.seq_len)
        self.last_l = int(core_model.patch_nums[-1] ** 2)
        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(core_model, "label_smoothing", 0.0)),
            reduction="none",
        )
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.loss_weight = torch.ones((1, self.seq_len), device=self.device, dtype=torch.float32) / float(self.seq_len)
        self._prog_it = 0
        self._last_prog_si = -1
        self._first_prog = True
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

    def _sync_vector(self, values: list[float]) -> list[float]:
        if not dist.is_available() or not dist.is_initialized():
            return values
        t = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return [float(x) for x in t.tolist()]

    def _compute_lr_ratio(self, step: int, total_steps: int) -> float:
        if self.schedule_name in {"none", ""}:
            return 1.0

        warmup_steps = self._warmup_steps
        if warmup_steps > 0 and step < warmup_steps:
            return self.lr_warmup_start_ratio + (1.0 - self.lr_warmup_start_ratio) * (
                float(step) / float(max(1, warmup_steps))
            )

        den = max(1, total_steps - 1 - warmup_steps)
        t = float(step - warmup_steps) / float(den)
        t = min(max(t, 0.0), 1.0)

        if self.schedule_name in {"cos", "cosine", "warmup_cosine"}:
            return self.final_lr_ratio + (1.0 - self.final_lr_ratio) * (0.5 + 0.5 * math.cos(math.pi * t))
        if self.schedule_name == "lin0":
            plateau_ratio = 0.05
            if t < plateau_ratio:
                return 1.0
            rest = (1.0 - t) / max(1e-8, 1.0 - plateau_ratio)
            return self.final_lr_ratio + (1.0 - self.final_lr_ratio) * rest
        if self.schedule_name == "lin":
            plateau_ratio = 0.15
            if t < plateau_ratio:
                return 1.0
            rest = (1.0 - t) / max(1e-8, 1.0 - plateau_ratio)
            return self.final_lr_ratio + (1.0 - self.final_lr_ratio) * rest
        return self.final_lr_ratio + (1.0 - self.final_lr_ratio) * (1.0 - t)

    def _set_lr_wd(self, step: int, total_steps: int):
        lr_ratio = self._compute_lr_ratio(step=step, total_steps=total_steps)
        for i, pg in enumerate(self.optimizer.param_groups):
            lr_sc = float(pg.get("lr_sc", 1.0))
            pg["lr"] = self.base_lrs[i] * lr_ratio * lr_sc

        if total_steps <= 1:
            pasd = 1.0
        else:
            pasd = float(step) / float(total_steps - 1)
        cur_wd = self.weight_decay_end + (self.weight_decay_start - self.weight_decay_end) * (
            0.5 + 0.5 * math.cos(math.pi * pasd)
        )
        for pg in self.optimizer.param_groups:
            if float(pg.get("weight_decay", 0.0)) > 0:
                wd_sc = float(pg.get("wd_sc", 1.0))
                pg["weight_decay"] = cur_wd * wd_sc

    def _compute_prog_si(self, global_step: int, total_steps: int) -> int:
        if self.progressive_ratio <= 0:
            return -1
        nstages = len(self.begin_ends)
        pg0 = min(max(self.progressive_start_stage, 0), nstages - 1)
        if global_step <= self._warmup_steps:
            si = pg0
        elif global_step >= total_steps * self.progressive_ratio:
            si = nstages - 1
        else:
            den = max(1.0, total_steps * self.progressive_ratio - self._warmup_steps)
            progress = float(global_step - self._warmup_steps) / float(den)
            progress = min(max(progress, 0.0), 1.0)
            delta = nstages - 1 - pg0
            si = pg0 + round(progress * delta)
        if si >= nstages - 1:
            return -1
        return si

    def _compute_prog_wp(self, prog_si: int, steps_per_epoch: int) -> float:
        if prog_si < 0:
            return 1.0
        if self._last_prog_si != prog_si:
            if self._last_prog_si != -1:
                self._first_prog = False
            self._last_prog_si = prog_si
            self._prog_it = 0
        self._prog_it += 1
        prog_wp_steps = max(1, int(round(self.progressive_warmup_epochs * steps_per_epoch)))
        prog_wp = max(min(self._prog_it / float(prog_wp_steps), 1.0), 0.01)
        if self._first_prog:
            prog_wp = 1.0
        return prog_wp

    def _step(self, ms_tokens: list[torch.Tensor], labels: torch.Tensor, prog_si: int, prog_wp: float):
        cond = None
        if self.tokenizer is not None:
            with torch.no_grad():
                cond = self.tokenizer.idx_to_var_input(ms_tokens)
        with autocast(enabled=self.amp):
            logits, targets = self.model(
                ms_tokens,
                cond_blc_wo_first_l=cond,
                class_labels=labels,
                prog_si=prog_si,
            )
            b, l, v = logits.shape
            loss = self.train_loss(logits.reshape(-1, v), targets.reshape(-1)).view(b, l)
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(float(prog_wp), 0.0), 1.0)
            else:
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        return loss

    def train_one_epoch(self, train_loader: DataLoader, epoch: int, total_steps: int, global_step: int):
        train_stats, global_step, _ = self.train_one_epoch_with_iterator(
            train_loader=train_loader,
            train_iter=iter(train_loader),
            epoch=epoch,
            total_steps=total_steps,
            global_step=global_step,
            steps_per_epoch=len(train_loader),
        )
        return train_stats, global_step

    def train_one_epoch_with_iterator(
        self,
        train_loader: DataLoader,
        train_iter,
        epoch: int,
        total_steps: int,
        global_step: int,
        steps_per_epoch: int,
    ):
        self.model.train()
        loss_sum = 0.0
        n = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"train {epoch}", leave=False, disable=not self.is_main_process)
        self.optimizer.zero_grad(set_to_none=True)

        for step_in_ep in pbar:
            try:
                ms_tokens, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                ms_tokens, labels = next(train_iter)
            ms_tokens = [t.to(self.device, non_blocking=True) for t in ms_tokens]
            labels = labels.to(self.device, non_blocking=True)
            self._set_lr_wd(step=global_step, total_steps=total_steps)
            prog_si = self._compute_prog_si(global_step=global_step, total_steps=total_steps)
            prog_wp = self._compute_prog_wp(prog_si=prog_si, steps_per_epoch=steps_per_epoch)

            stepping = ((step_in_ep + 1) % self.grad_accum_steps) == 0
            if hasattr(self.model, "require_backward_grad_sync"):
                self.model.require_backward_grad_sync = stepping

            loss = self._step(ms_tokens, labels, prog_si=prog_si, prog_wp=prog_wp)
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            if stepping:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            bs = ms_tokens[0].shape[0]
            loss_sum += float(loss.detach()) * bs
            n += bs
            pbar.set_postfix(loss=f"{loss_sum / max(1, n):.4f}")
            global_step += 1

        rem = steps_per_epoch % self.grad_accum_steps
        if rem != 0:
            if hasattr(self.model, "require_backward_grad_sync"):
                self.model.require_backward_grad_sync = True
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)

        loss_sum, n = self._sync(loss_sum, n)
        return {"loss": loss_sum / max(1, n)}, global_step, train_iter

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int):
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        try:
            self.model.eval()
            loss_sum = 0.0
            loss_tail_sum = 0.0
            acc_sum = 0.0
            acc_tail_sum = 0.0
            n = 0
            pbar = tqdm(val_loader, desc=f"val {epoch}", leave=False, disable=not self.is_main_process)

            for ms_tokens, labels in pbar:
                ms_tokens = [t.to(self.device, non_blocking=True) for t in ms_tokens]
                labels = labels.to(self.device, non_blocking=True)
                cond = None
                if self.tokenizer is not None:
                    cond = self.tokenizer.idx_to_var_input(ms_tokens)
                logits, targets = self.model(
                    ms_tokens,
                    cond_blc_wo_first_l=cond,
                    class_labels=labels,
                    prog_si=-1,
                )
                b, _, v = logits.shape
                loss = self.val_loss(logits.reshape(-1, v), targets.reshape(-1))
                loss_tail = self.val_loss(logits[:, -self.last_l:, :].reshape(-1, v), targets[:, -self.last_l:].reshape(-1))
                pred = logits.argmax(dim=-1)
                acc = (pred == targets).float().mean().item() * 100.0
                acc_tail = (pred[:, -self.last_l:] == targets[:, -self.last_l:]).float().mean().item() * 100.0
                bs = ms_tokens[0].shape[0]
                loss_sum += float(loss.detach()) * bs
                loss_tail_sum += float(loss_tail.detach()) * bs
                acc_sum += float(acc) * bs
                acc_tail_sum += float(acc_tail) * bs
                n += bs

            loss_sum, loss_tail_sum, acc_sum, acc_tail_sum, n = self._sync_vector(
                [loss_sum, loss_tail_sum, acc_sum, acc_tail_sum, float(n)]
            )
            n = int(n)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        return {
            "loss": loss_sum / max(1, n),
            "loss_tail": loss_tail_sum / max(1, n),
            "acc": acc_sum / max(1, n),
            "acc_tail": acc_tail_sum / max(1, n),
        }

    def _generate_and_save_samples(self, epoch: int):
        """Generate a small batch of images and save them for visual validation."""
        if not self.is_main_process or self.sample_fn is None:
            return
        sample_dir = self.save_dir / "samples" / f"epoch_{epoch:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        model = self._unwrap_model()
        was_training = model.training
        model.eval()

        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        try:
            with torch.no_grad():
                self.sample_fn(model, sample_dir, self.num_val_samples)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
            if was_training:
                model.train()

        self._log(f"[epoch {epoch}] saved {self.num_val_samples} samples -> {sample_dir}")

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
        if self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()
        torch.save(ckpt, self.save_dir / "last.pt")
        if best:
            torch.save(ckpt, self.save_dir / "best.pt")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        steps_per_epoch: int | None = None,
        eval_every: int = 1,
        save_every: int = 1,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        sample_every: int = 0,
    ):
        best_val = float("inf")
        best_epoch = -1
        no_improve_count = 0
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)
        steps_per_epoch = int(steps_per_epoch)
        total_steps = max(1, epochs * steps_per_epoch)
        self._warmup_steps = int(round(self.warmup_epochs * steps_per_epoch))
        global_step = 0
        train_iter = iter(train_loader)

        for epoch in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_stats, global_step, train_iter = self.train_one_epoch_with_iterator(
                train_loader=train_loader,
                train_iter=train_iter,
                epoch=epoch,
                total_steps=total_steps,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
            )

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

            if save_every > 0 and epoch % save_every == 0:
                self.save_checkpoint(epoch, best=False)
                self._log(f"[epoch {epoch}] saved last.pt")

            if sample_every > 0 and epoch % sample_every == 0:
                self._generate_and_save_samples(epoch)

            if val_stats is None:
                self._log(f"[epoch {epoch}] train loss={train_stats['loss']:.4f}")
            else:
                self._log(
                    f"[epoch {epoch}] train loss={train_stats['loss']:.4f} | "
                    f"val loss={val_stats['loss']:.4f} (tail={val_stats['loss_tail']:.4f}) | "
                    f"acc={val_stats['acc']:.2f} tail_acc={val_stats['acc_tail']:.2f}"
                )

            if early_stopping_patience > 0 and val_stats is not None and no_improve_count >= early_stopping_patience:
                self._log(
                    f"[epoch {epoch}] early stopping triggered: no improvement for {no_improve_count} evals "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
                )
                break
