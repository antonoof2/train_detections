from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path(img_path: Path) -> Path:
    """YOLO-соглашение: .../images/x.jpg -> .../labels/x.txt (иначе txt лежит рядом с картинкой)."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")

    return img_path.with_suffix(".txt")


def read_yolo(path: Path, w: int, h: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Читает YOLO-разметку, возвращает боксы XYXY в пикселях и классы. Нет файла — кадр без объектов."""
    boxes, labels = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if not 0 <= cls < num_classes:
                raise ValueError(f"{path}: класс {cls} вне диапазона 0..{num_classes - 1}")

            vals = np.array(parts[1:], dtype=np.float32)
            if len(vals) <= 5:                       # cx cy w h (лишнее значение — confidence — игнорируем)
                cx, cy, bw, bh = vals[:4]
            else:                                    # полигон (YOLO-seg): берём охватывающий бокс
                xs, ys = vals[0::2], vals[1::2]
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                bw, bh = xs.max() - xs.min(), ys.max() - ys.min()

            x1, y1 = max((cx - bw / 2) * w, 0.0), max((cy - bh / 2) * h, 0.0)
            x2, y2 = min((cx + bw / 2) * w, float(w)), min((cy + bh / 2) * h, float(h))
            if x2 - x1 > 0 and y2 - y1 > 0:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def build_transforms(size: int, train: bool, strong: bool) -> T.Compose:
    """Аугментации как в оригинальном D-FINE (torchvision v2); без нормализации — только деление на 255."""
    ops = []
    if train:
        if strong:
            ops += [
                T.RandomPhotometricDistort(p=0.5),
                T.RandomZoomOut(fill=0, side_range=(1.0, 2.0), p=0.3),
                T.RandomApply([T.RandomIoUCrop()], p=0.8),
                T.SanitizeBoundingBoxes(min_size=1),
            ]
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [
        T.Resize((size, size)),
        T.SanitizeBoundingBoxes(min_size=1),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]

    return T.Compose(ops)


class YoloDataset(Dataset):
    """Датасет в YOLO-формате. Отдаёт тензор 3xSxS и target в формате HF D-FINE:
    class_labels (N,) и boxes (N, 4) — cx, cy, w, h в долях от размера кадра."""

    def __init__(self, images_dir: Path, size: int, num_classes: int, train: bool):
        if not images_dir.exists():
            raise FileNotFoundError(f"Папка с картинками не найдена: {images_dir}")
        self.images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not self.images:
            raise FileNotFoundError(f"В {images_dir} нет картинок {sorted(IMG_EXTS)}")

        self.size = size
        self.num_classes = num_classes
        self.train = train
        self.set_strong_aug(True)

    def set_strong_aug(self, strong: bool) -> None:
        self.transforms = build_transforms(self.size, self.train, strong)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        path = self.images[i]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        boxes, labels = read_yolo(label_path(path), w, h, self.num_classes)

        target = {
            "boxes": tv_tensors.BoundingBoxes(torch.from_numpy(boxes), format="XYXY", canvas_size=(h, w)),
            "labels": torch.from_numpy(labels),
        }
        img, target = self.transforms(img, target)

        x1, y1, x2, y2 = target["boxes"].as_subclass(torch.Tensor).unbind(-1)
        cxcywh = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1) / self.size

        return img, {"class_labels": target["labels"].long(), "boxes": cxcywh.clamp(0, 1).float()}


def collate_fn(batch):
    images, targets = zip(*batch)

    return torch.stack(images), list(targets)


def make_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    root = Path(cfg.data.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Датасет не найден: {root}")

    t = cfg.train
    if t.resolution % 32:
        raise ValueError(f"train.resolution должен быть кратен 32, а не {t.resolution}")

    num_classes = len(cfg.data.names)
    train_ds = YoloDataset(root / cfg.data.train, t.resolution, num_classes, train=True)
    val_ds = YoloDataset(root / cfg.data.val, t.resolution, num_classes, train=False)
    print(f"train: {len(train_ds)} кадров, val: {len(val_ds)} кадров")

    common = dict(num_workers=t.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds, batch_size=t.batch_size, shuffle=True, drop_last=len(train_ds) > t.batch_size, **common
    )
    val_loader = DataLoader(val_ds, batch_size=t.batch_size, shuffle=False, **common)

    return train_loader, val_loader


def pick_samples(data_cfg: DictConfig, samples_cfg: DictConfig) -> list[Path]:
    """Фиксированные кадры из конфига для визуального сравнения моделей."""
    root = Path(data_cfg.root).expanduser().resolve()
    images = [root / p for p in samples_cfg.images]

    missing = [p for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Кадры для samples не найдены: {missing}")

    return images
