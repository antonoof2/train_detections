import cv2
import supervision as sv

from pathlib import Path
from clearml import Task
from rfdetr import RFDETR
from omegaconf import DictConfig

BOX = sv.BoxAnnotator()
LABEL = sv.LabelAnnotator()


def draw(model: RFDETR, frame_bgr, conf: float):
    """Предсказание на одном BGR-кадре, возвращает BGR-кадр с боксами."""
    detections = model.predict(frame_bgr[:, :, ::-1].copy(), threshold=conf, include_source_image=False)
    names = model.class_names
    labels = [f"{names[c] if c < len(names) else c} {s:.2f}"
              for c, s in zip(detections.class_id, detections.confidence)]
    frame = BOX.annotate(frame_bgr.copy(), detections)

    return LABEL.annotate(frame, detections, labels=labels)


def validate(model: RFDETR, cfg: DictConfig, task: Task) -> None:
    """Считает метрики на val и отправляет их в ClearML."""
    metrics = model.evaluate(
        split="val",
        dataset_dir=str(Path(cfg.data.root).expanduser().resolve()),
        resolution=cfg.train.resolution,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        device=cfg.train.device,
        log_per_class_metrics=True,
    )
    logger = task.get_logger()

    # Общие метрики по всем классам — те же имена, что были у YOLO, чтобы сравнивать в ClearML
    summary = {
        "precision": metrics.get("val/precision"),
        "recall": metrics.get("val/recall"),
        "mAP50": metrics.get("val/mAP_50"),
        "mAP50-95": metrics.get("val/mAP_50_95"),
        "F1": metrics.get("val/F1"),
    }
    for name, value in summary.items():
        if value is None:
            continue
        logger.report_single_value(f"val/{name}", float(value))
        print(f"val/{name}: {value:.4f}")

    # AP по каждому классу (первая строка — заголовок таблицы)
    rows: list[list] = [["class", "mAP50-95"]]
    for name in cfg.data.names:
        if (ap := metrics.get(f"val/AP/{name}")) is not None:
            rows.append([name, round(ap, 4)])
    logger.report_table("Validation", "per-class metrics", iteration=0, table_plot=rows)


def log_samples(model: RFDETR, images: list[Path], cfg: DictConfig, task: Task) -> None:
    """Кладёт предсказания на фиксированных кадрах в ClearML Debug Samples (title/series общие для всех моделей)."""
    logger = task.get_logger()
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"не смог прочитать {path}, пропускаю")
            continue
        annotated = draw(model, frame, cfg.predict.conf)
        logger.report_image("samples", path.name, iteration=0, image=annotated[:, :, ::-1])


def predict_video(model: RFDETR, video_path: Path, out_path: Path, conf: float) -> Path:
    """Прогоняет модель по видео и сохраняет её с нарисованными боксами."""
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
            writer.write(draw(model, frame, conf))
    finally:
        cap.release()
        writer.release()

    return out_path
