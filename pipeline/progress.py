import time

from tqdm.auto import tqdm


def make_epoch_bar(total_epochs: int):
    """Колбэк Lightning: полоса по эпохам — сколько прошло, сколько осталось."""
    from pytorch_lightning import Callback

    class EpochProgress(Callback):
        def __init__(self, total):
            super().__init__()
            self.total = total
            self.bar = None
            self.t0 = None

        def on_train_start(self, trainer, pl_module):
            self.t0 = time.time()
            self.bar = tqdm(
                total=self.total, desc="эпохи", unit="эп",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[прошло {elapsed}, осталось {remaining}{postfix}]",
                dynamic_ncols=True, leave=True, position=0,
            )

        def on_train_epoch_end(self, trainer, pl_module):
            if self.bar is None:
                return
            post = {}
            for key, val in trainer.callback_metrics.items():
                k = str(key).lower()
                if "map" in k and "ema" not in k and len(post) < 2:
                    try:
                        post[str(key).split("/")[-1]] = f"{float(val):.3f}"
                    except (TypeError, ValueError):
                        pass
            if post:
                self.bar.set_postfix(post)
            self.bar.update(1)

        def on_train_end(self, trainer, pl_module):
            if self.bar is not None:
                self.bar.close()
                self.bar = None
            if self.t0:
                print(f"обучение заняло {(time.time() - self.t0) / 60:.1f} мин")

    return EpochProgress(total_epochs)


def attach_epoch_bar(total_epochs: int) -> None:
    """RF-DETR не принимает callbacks в train(), поэтому оборачиваем build_trainer.

    Импорт build_trainer внутри train() ленивый, так что подмена атрибута модуля срабатывает.
    """
    import rfdetr.training as rt

    original = rt.build_trainer

    def wrapped(*args, **kwargs):
        trainer = original(*args, **kwargs)
        try:
            trainer.callbacks.append(make_epoch_bar(total_epochs))
        except Exception as e:
            print("не смог повесить полосу прогресса по эпохам:", e)
        
        return trainer

    rt.build_trainer = wrapped
