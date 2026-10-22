import copy
import math
import re
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import DFineForObjectDetection

SIZES = ("nano", "small", "medium", "large", "xlarge")


def resolve_source(model_cfg: DictConfig) -> str:
    """Откуда грузить веса: COCO с HF Hub (по умолчанию), id с HF Hub или локальная папка."""
    weights = model_cfg.weights
    if not weights:
        if model_cfg.size not in SIZES:
            raise ValueError(f"model.size должен быть одним из {SIZES}, а не {model_cfg.size!r}")
        return f"ustc-community/dfine-{model_cfg.size}-coco"

    weights = str(weights)
    if weights.endswith((".pth", ".pt", ".ckpt")):
        raise ValueError("model.weights — папка с чекпоинтом (save_pretrained) или id с HF Hub, а не файл .pth")
    if Path(weights).exists():
        return str(Path(weights).resolve())
    if re.fullmatch(r"[\w.-]+/[\w.-]+", weights):   # похоже на id репозитория, например ustc-community/...
        return weights

    raise FileNotFoundError(f"model.weights не найден: {weights}")


def build_model(model_cfg: DictConfig, names: list[str], device: torch.device) -> DFineForObjectDetection:
    """Загружает предобученный D-FINE и заменяет классификационные головы под наши классы."""
    id2label = dict(enumerate(names))
    model = DFineForObjectDetection.from_pretrained(
        resolve_source(model_cfg),
        num_labels=len(names),
        id2label=id2label,
        label2id={n: i for i, n in id2label.items()},
        ignore_mismatched_sizes=True,   # головы под 80 классов COCO пересоздаются под наши
    )

    return model.to(device)


def load_model(path: Path, device: torch.device) -> DFineForObjectDetection:
    """Загружает обученный чекпоинт (папка, сохранённая через save_pretrained)."""
    return DFineForObjectDetection.from_pretrained(str(path)).to(device).eval()


class ModelEMA:
    """EMA весов с разогревом: decay_t = decay * (1 - exp(-t / tau)), как в оригинальном D-FINE."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1 - d)
            else:
                v.copy_(src[k])


def build_optimizer(model: nn.Module, lr: float, lr_backbone: float, weight_decay: float) -> AdamW:
    """AdamW с отдельным lr для бэкбона; weight decay не применяется к bias и нормализациям."""
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = ".backbone." in f".{name}"
        no_decay = p.ndim <= 1
        groups.setdefault((is_backbone, no_decay), []).append(p)

    param_groups = [
        {"params": ps, "lr": lr_backbone if bb else lr, "weight_decay": 0.0 if nd else weight_decay}
        for (bb, nd), ps in groups.items()
    ]

    return AdamW(param_groups, betas=(0.9, 0.999))


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> LambdaLR:
    """Линейный разогрев, затем косинус до lr * min_lr_ratio. Шаг — на каждый шаг оптимизатора."""
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    return LambdaLR(optimizer, factor)
