import cv2
import numpy as np
import supervision as sv
import torch

from pathlib import Path
from clearml import Task
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from pipeline.evaluate import evaluate
from pipeline.inference import Detector
from pipeline.utils import normalize_amp

BOX = sv.BoxAnnotator()
LABEL = sv.LabelAnnotator()


def draw(detector: Detector, frame_bgr: np.ndarray, conf: float, names: list[str]) -> np.ndarray:
    """Предсказание на одном BGR-кадре, возвращает BGR-кадр с боксами."""
    xyxy, scores, labels = detector(frame_bgr, conf)
    detections = sv.Detections(
        xyxy=xyxy.reshape(-1, 4).astype(np.float32),
        confidence=scores.astype(np.float32),
        class_id=labels.astype(int),
    )
    texts = [f"{names[c] if c < len(names) else c} {s:.2f}" for c, s in zip(labels, scores)]
    frame = BOX.annotate(frame_bgr.copy(), detections)

    return LABEL.annotate(frame, detections, labels=texts)


def validate(model: nn.Module, val_loader: DataLoader, cfg: DictConfig, task: Task, device: torch.device) -> None:
    """Считает метрики на val и отправляет их в ClearML."""
    names = list(cfg.data.names)
    metrics = evaluate(model, val_loader, device, names, cfg.predict.conf, normalize_amp(cfg.train.amp, device))
    logger = task.get_logger()

    # Общие метрики по всем классам — те же имена, что у YOLO и RF-DETR, чтобы сравнивать в ClearML
    for name in ("precision", "recall", "mAP50", "mAP50-95", "F1"):
        logger.report_single_value(f"val/{name}", float(metrics[name]))
        print(f"val/{name}: {metrics[name]:.4f}")

    # AP по каждому классу (первая строка — заголовок таблицы)
    rows: list[list] = [["class", "mAP50-95"]]
    for name in names:
        if (ap := metrics.get(f"AP/{name}")) is not None:
            rows.append([name, round(ap, 4)])
    logger.report_table("Validation", "per-class metrics", iteration=0, table_plot=rows)


def log_samples(detector: Detector, images: list[Path], cfg: DictConfig, task: Task) -> None:
    """Кладёт предсказания на фиксированных кадрах в ClearML Debug Samples (title/series общие для всех моделей)."""
    logger = task.get_logger()
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"не смог прочитать {path}, пропускаю")
            continue
        annotated = draw(detector, frame, cfg.predict.conf, list(cfg.data.names))
        logger.report_image("samples", path.name, iteration=0, image=annotated[:, :, ::-1])


def predict_video(detector: Detector, video_path: Path, out_path: Path, conf: float, names: list[str]) -> Path:
    """Прогоняет модель по видео и сохраняет его с нарисованными боксами."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        # кадры обрабатываются по одному, без накопления в памяти
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(draw(detector, frame, conf, names))
    finally:
        cap.release()
        writer.release()

    return out_path
