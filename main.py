import gc
import hydra
import torch

from pathlib import Path
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from pipeline.data import make_loaders, pick_samples
from pipeline.inference import Detector
from pipeline.model import build_model, load_model
from pipeline.report import log_samples, predict_video, validate
from pipeline.train import fit
from pipeline.utils import normalize_amp, seed_everything


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    task = Task.init(
        project_name=cfg.clearml.project_name,
        task_name=cfg.clearml.task_name,
        tags=list(cfg.clearml.tags),
    )
    print(OmegaConf.to_yaml(cfg))

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    device = torch.device(cfg.train.device)
    seed_everything(cfg.train.seed)
    names = list(cfg.data.names)

    train_loader, val_loader = make_loaders(cfg)

    # Обучение (weights=null — старт с COCO, иначе — с переданного чекпоинта, например stage1)
    model = build_model(cfg.model, names, device)
    best = fit(model, train_loader, val_loader, cfg, device, out_dir / "weights", task)
    task.upload_artifact("best_weights", artifact_object=str(best))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Валидация лучших весов (в памяти после обучения — веса последней эпохи, поэтому грузим best с диска)
    best_model = load_model(best, device)
    validate(best_model, val_loader, cfg, task, device)

    # Фиксированные кадры и видео (чтобы в ClearML потом смотреть различия между моделями)
    detector = Detector(best_model, cfg.train.resolution, device, normalize_amp(cfg.train.amp, device))
    log_samples(detector, pick_samples(cfg.data, cfg.samples), cfg, task)
    if cfg.predict.video:
        video_path = Path(cfg.predict.video)
        out_path = predict_video(detector, video_path, out_dir / f"{video_path.stem}_pred.mp4", cfg.predict.conf, names)
        task.upload_artifact("predicted_video", artifact_object=str(out_path))
        print(f"Видео с разметкой сохранено: {out_path}")

    task.close()


if __name__ == "__main__":
    main()
