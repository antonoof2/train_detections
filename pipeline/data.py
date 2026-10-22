import yaml

from pathlib import Path
from omegaconf import DictConfig


def make_data_yaml(data_cfg: DictConfig) -> Path:
    """Пишет data.yaml в корень датасета: RF-DETR ищет его только там.

    Пути в нём относительные — RF-DETR не пускает сплиты за пределы root.
    """
    root = Path(data_cfg.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Датасет не найден: {root}")

    data = {
        "train": data_cfg.train,
        "val": data_cfg.val,
        "test": data_cfg.val,
        "nc": len(data_cfg.names),
        "names": dict(enumerate(data_cfg.names)),
    }

    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return data_yaml


def pick_samples(data_cfg: DictConfig, samples_cfg: DictConfig) -> list[Path]:
    """Фиксированные кадры из конфига для визуального сравнения моделей."""
    root = Path(data_cfg.root).expanduser().resolve()
    images = [root / p for p in samples_cfg.images]

    missing = [p for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Кадры для samples не найдены: {missing}")

    return images
