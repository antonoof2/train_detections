import random

import numpy as np
import torch

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_amp(mode, device: torch.device) -> str:
    """Приводит значение amp из конфига к 'bf16' | 'fp16' | 'none'.

    YAML 1.1 превращает голые off/no в False, поэтому здесь принимаются и они.
    bf16 на GPU без его поддержки (до Ampere) заменяется на fp16.
    """
    mode = str(mode).lower()
    if mode not in AMP_DTYPES:
        return "none"
    if mode == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return "fp16"
    return mode


def autocast(device: torch.device, mode: str):
    """Контекст смешанной точности; при mode='none' — пустышка."""
    dtype = AMP_DTYPES.get(mode)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)
