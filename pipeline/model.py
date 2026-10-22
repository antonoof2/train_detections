import copy
import json
import math
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Код модели и loss — из официального репозитория DEIMv2 (клонируется в third_party через `make install`)
DEIMV2_DIR = Path(__file__).resolve().parents[1] / "third_party" / "DEIMv2"
if not (DEIMV2_DIR / "engine").exists():
    raise ImportError(f"Не найден код DEIMv2 в {DEIMV2_DIR} — выполните `make install`")
sys.path.insert(0, str(DEIMV2_DIR))

from engine.backbone import DINOv3STAs, HGNetv2  # noqa: E402
from engine.deim import DEIMCriterion, DEIMTransformer, HungarianMatcher, HybridEncoder, LiteEncoder  # noqa: E402

# COCO-веса DEIMv2 на HF Hub: atto/femto/pico/nano — бэкбон HGNetv2, small..xlarge — DINOv3
HUB_REPOS = {
    "atto": "Intellindust/DEIMv2_HGNetv2_ATTO_COCO",
    "femto": "Intellindust/DEIMv2_HGNetv2_FEMTO_COCO",
    "pico": "Intellindust/DEIMv2_HGNetv2_PICO_COCO",
    "nano": "Intellindust/DEIMv2_HGNetv2_N_COCO",
    "small": "Intellindust/DEIMv2_DINOv3_S_COCO",
    "medium": "Intellindust/DEIMv2_DINOv3_M_COCO",
    "large": "Intellindust/DEIMv2_DINOv3_L_COCO",
    "xlarge": "Intellindust/DEIMv2_DINOv3_X_COCO",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sync_bn_to_bn(module: nn.Module) -> None:
    """SyncBatchNorm в адаптере DINOv3 -> обычный BatchNorm2d (обучение на одной GPU, без DDP)."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            setattr(module, name, nn.BatchNorm2d(
                child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats
            ))
        else:
            sync_bn_to_bn(child)


class DEIMv2(nn.Module, PyTorchModelHubMixin):
    """DEIMv2 (бэкбон + энкодер + декодер) в формате HF Hub: save_pretrained / from_pretrained.

    config — словарь как в config.json на HF Hub (ключи — классы модулей из engine).
    На вход — RGB в [0, 1]; для DINOv3-вариантов ImageNet-нормализация делается внутри модели.
    """

    def __init__(self, config: dict):
        super().__init__()
        if "DINOv3STAs" in config:
            self.backbone = DINOv3STAs(**config["DINOv3STAs"])
        else:
            self.backbone = HGNetv2(**config["HGNetv2"])
        if "LiteEncoder" in config:
            self.encoder = LiteEncoder(**config["LiteEncoder"])
        else:
            self.encoder = HybridEncoder(**config["HybridEncoder"])
        self.decoder = DEIMTransformer(**config["DEIMTransformer"])
        sync_bn_to_bn(self)

        self.normalize = "DINOv3STAs" in config
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, targets: list[dict] | None = None) -> dict:
        """В train-режиме нужны targets (denoising-запросы), в eval — нет. Возвращает pred_logits, pred_boxes (+aux)."""
        if self.normalize:
            x = (x - self.mean) / self.std

        return self.decoder(self.encoder(self.backbone(x)), targets)


def resolve_source(model_cfg: DictConfig) -> str:
    """Откуда грузить веса: COCO с HF Hub (по умолчанию), id с HF Hub или локальная папка."""
    weights = model_cfg.weights
    if not weights:
        if model_cfg.size not in HUB_REPOS:
            raise ValueError(f"model.size должен быть одним из {tuple(HUB_REPOS)}, а не {model_cfg.size!r}")
        return HUB_REPOS[model_cfg.size]

    weights = str(weights)
    if weights.endswith((".pth", ".pt", ".ckpt")):
        raise ValueError("model.weights — папка с чекпоинтом (save_pretrained) или id с HF Hub, а не файл .pth")
    if Path(weights).exists():
        return str(Path(weights).resolve())
    if re.fullmatch(r"[\w.-]+/[\w.-]+", weights):   # похоже на id репозитория, например Intellindust/...
        return weights

    raise FileNotFoundError(f"model.weights не найден: {weights}")


def fetch(source: str, filename: str) -> Path:
    """Файл чекпоинта из локальной папки или с HF Hub."""
    if Path(source).is_dir():
        return Path(source) / filename
    return Path(hf_hub_download(source, filename))


def build_model(model_cfg: DictConfig, names: list[str], resolution: int, device: torch.device) -> DEIMv2:
    """Загружает предобученный DEIMv2 и пересоздаёт классификационные головы под наши классы."""
    source = resolve_source(model_cfg)
    config = copy.deepcopy(json.loads(fetch(source, "config.json").read_text()))
    config["names"] = list(names)
    config["DEIMTransformer"]["num_classes"] = len(names)
    # якоря и позиционные эмбеддинги для eval считаются заранее под этот размер — он должен совпадать с инференсом
    for key in ("HybridEncoder", "LiteEncoder", "DEIMTransformer"):
        if key in config:
            config[key]["eval_spatial_size"] = [resolution, resolution]

    model = DEIMv2(config)
    own = model.state_dict()
    state = load_file(str(fetch(source, "model.safetensors")))
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(k for k in state if k not in matched)
    model.load_state_dict(matched, strict=False)
    print(f"веса из {source}: загружено {len(matched)} тензоров, пересоздано {len(skipped)} (головы под наши классы)")

    return model.to(device)


def load_model(path: Path, device: torch.device) -> DEIMv2:
    """Загружает обученный чекпоинт (папка, сохранённая через save_pretrained)."""
    return DEIMv2.from_pretrained(str(path)).to(device).eval()


def build_criterion(num_classes: int, epochs: int) -> DEIMCriterion:
    """Loss DEIMv2 (configs/base/deimv2.yml): MAL + L1/GIoU + FGL/DDF, matcher с IoU-порядком на поздних эпохах."""
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
        change_matcher=True,
        iou_order_alpha=4.0,
        matcher_change_epoch=int(epochs * 0.8),   # в оригинале ~80% эпох (45 из 58, 50 из 68)
    )

    return DEIMCriterion(
        matcher,
        weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2, "loss_fgl": 0.15, "loss_ddf": 1.5},
        losses=["mal", "boxes", "local"],
        alpha=0.75,
        gamma=1.5,
        num_classes=num_classes,
        reg_max=32,
    )


class ModelEMA:
    """EMA весов с разогревом: decay_t = decay * (1 - exp(-t / tau)), как в оригинальном DEIM."""

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
    """AdamW с отдельным lr для бэкбона; weight decay не применяется к bias и нормализациям.

    У DINOv3-вариантов малый lr только у самого ViT (backbone.dinov3), STA-адаптер учится с основным — как в DEIMv2.
    """
    names = [n for n, _ in model.named_parameters()]
    backbone_key = ".dinov3." if any(".dinov3." in f".{n}" for n in names) else ".backbone."

    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = backbone_key in f".{name}"
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
