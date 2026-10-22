import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from clearml import Task
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipeline.evaluate import evaluate
from pipeline.model import ModelEMA, build_criterion, build_optimizer, build_scheduler
from pipeline.progress import EpochBar, batch_bar
from pipeline.utils import autocast, normalize_amp

# Составляющие loss DEIMv2: matchability-aware, L1, GIoU, fine-grained localization, decoupled distillation focal
LOSS_PARTS = ("loss_mal", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
VAL_SCALARS = ("mAP50", "mAP50-95", "precision", "recall", "F1")


def multiscale_sizes(resolution: int, span: int) -> list[int]:
    return [s for s in range(resolution - span, resolution + span + 1, 32) if s >= 224]


def to_float32(x):
    """Выходы модели после autocast -> fp32 (loss DEIMv2 считается без autocast, как в оригинале)."""
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: to_float32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_float32(v) for v in x)
    return x


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    weights_dir: Path,
    task: Task,
) -> Path:
    """Обучает модель, после каждой эпохи считает метрики на val, сохраняет best/last (по mAP50-95).

    Метрики и чекпоинты берутся с EMA-весов (если train.ema=true). Возвращает папку с лучшим чекпоинтом.
    """
    t = cfg.train
    names = list(cfg.data.names)
    logger = task.get_logger()
    amp = normalize_amp(t.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16" and device.type == "cuda")
    accum = t.grad_accum_steps

    criterion = build_criterion(len(names), t.epochs)
    ema = ModelEMA(model, t.ema_decay, t.ema_tau) if t.ema else None
    optimizer = build_optimizer(model, t.lr, t.lr_backbone, t.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    scheduler = build_scheduler(optimizer, steps_per_epoch * t.epochs, t.warmup_steps, t.min_lr_ratio)
    sizes = multiscale_sizes(t.resolution, t.multiscale_range) if t.multiscale else None
    print(f"amp={amp}, ema={bool(ema)}, шагов оптимизатора за эпоху: {steps_per_epoch}, multiscale: {sizes}")

    best_map, best_epoch, global_step = -1.0, 0, 0
    bar = EpochBar(t.epochs)
    for epoch in range(1, t.epochs + 1):
        strong = epoch <= t.epochs - t.no_aug_last_epochs     # в последних эпохах — без тяжёлых аугментаций
        train_loader.dataset.set_strong_aug(strong)             # воркеры пересоздаются каждую эпоху и подхватят флаг

        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, n_loss, skipped = 0.0, 0, 0
        parts: dict[str, torch.Tensor] = {}

        for i, (images, targets) in enumerate(batch_bar(train_loader, epoch, t.epochs), start=1):
            images = images.to(device, non_blocking=True)
            if sizes and strong:                                # размер в долях не влияет на боксы (они нормированы)
                s = random.choice(sizes)
                images = F.interpolate(images, size=(s, s), mode="bilinear", align_corners=False)
            labels = [{k: v.to(device, non_blocking=True) for k, v in tg.items()} for tg in targets]

            with autocast(device, amp):
                out = model(images, targets=labels)
            loss_dict = criterion(to_float32(out), labels, epoch=epoch - 1)   # matcher считает эпохи с нуля
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss / accum).backward()
            loss_sum += loss.item()
            n_loss += 1
            for key, val in loss_dict.items():
                for part in LOSS_PARTS:
                    if key.startswith(part):
                        parts[part] = parts.get(part, 0.0) + val.detach().float()

            if i % accum == 0 or i == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema:
                    ema.update(model)

                global_step += 1
                if global_step % t.log_every == 0:
                    logger.report_scalar("train_step", "loss", loss.item(), global_step)
                    logger.report_scalar("train_step", "lr", max(scheduler.get_last_lr()), global_step)

        if skipped:
            tqdm.write(f"эпоха {epoch}: пропущено {skipped} батчей с неконечным loss")
        train_loss = loss_sum / max(n_loss, 1)
        logger.report_scalar("loss", "train", train_loss, epoch)
        for part, val in parts.items():
            logger.report_scalar("loss_parts", part.removeprefix("loss_"), float(val) / max(n_loss, 1), epoch)
        lrs = scheduler.get_last_lr()
        logger.report_scalar("lr", "head", max(lrs), epoch)
        logger.report_scalar("lr", "backbone", min(lrs), epoch)

        eval_model = ema.module if ema else model
        postfix = {"loss": train_loss}
        evaluated = epoch % t.eval_interval == 0 or epoch == t.epochs
        if evaluated:
            metrics = evaluate(eval_model, val_loader, device, names, cfg.predict.conf, amp)
            for key in VAL_SCALARS:
                logger.report_scalar("val", key, metrics[key], epoch)
            postfix["mAP50-95"] = metrics["mAP50-95"]
            if metrics["mAP50-95"] > best_map:
                best_map, best_epoch = metrics["mAP50-95"], epoch
                eval_model.save_pretrained(weights_dir / "best")
                tqdm.write(f"эпоха {epoch}: новый лучший mAP50-95 = {best_map:.4f}")
        eval_model.save_pretrained(weights_dir / "last")
        bar.update(**postfix)

        if evaluated and t.early_stopping and epoch - best_epoch >= t.early_stopping_patience:
            tqdm.write(f"ранняя остановка: mAP50-95 не растёт {t.early_stopping_patience} эпох (лучшая: {best_epoch})")
            break
    bar.close()

    return weights_dir / "best"
