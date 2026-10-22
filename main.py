import hydra

from pathlib import Path
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from rfdetr import RFDETR, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from pipeline.data import make_data_yaml, pick_samples
from pipeline.progress import attach_epoch_bar
from pipeline.report import log_samples, predict_video, validate

MODELS = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
BEST_CHECKPOINTS = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    task = Task.init(
        project_name=cfg.clearml.project_name,
        task_name=cfg.clearml.task_name,
        tags=list(cfg.clearml.tags),
    )
    print(OmegaConf.to_yaml(cfg))

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    data_yaml = make_data_yaml(cfg.data)
    task.upload_artifact("data_yaml", artifact_object=str(data_yaml))

    # Обучение (weights=null — старт с COCO, иначе — с переданного чекпоинта, например stage1)
    model_kwargs = {"pretrain_weights": cfg.model.weights} if cfg.model.weights else {}
    model = MODELS[cfg.model.size](**model_kwargs)
    attach_epoch_bar(cfg.train.epochs)
    weights_dir = out_dir / "weights"
    model.train(
        dataset_dir=str(data_yaml.parent),
        output_dir=str(weights_dir),
        **OmegaConf.to_container(cfg.train, resolve=True),
    )

    # Валидация лучших весов (после train() в памяти веса последней эпохи)
    best = next(weights_dir / n for n in BEST_CHECKPOINTS if (weights_dir / n).exists())
    task.upload_artifact("best_weights", artifact_object=str(best))
    best_model = RFDETR.from_checkpoint(best)
    validate(best_model, cfg, task)

    # Фиксированные кадры и видео (чтобы в ClearML потом смотреть различия между моделями)
    log_samples(best_model, pick_samples(cfg.data, cfg.samples), cfg, task)
    if cfg.predict.video:
        video_path = Path(cfg.predict.video)
        out_path = predict_video(best_model, video_path, out_dir / f"{video_path.stem}_pred.mp4", cfg.predict.conf)
        task.upload_artifact("predicted_video", artifact_object=str(out_path))
        print(f"Видео с разметкой сохранено: {out_path}")

    task.close()


if __name__ == "__main__":
    main()
