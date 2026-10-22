import time

from tqdm.auto import tqdm


class EpochBar:
    """Полоса по эпохам — сколько прошло, сколько осталось; внутри эпохи — отдельная полоса по батчам."""

    def __init__(self, total_epochs: int):
        self.t0 = time.time()
        self.bar = tqdm(
            total=total_epochs, desc="эпохи", unit="эп",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [прошло {elapsed}, осталось {remaining}{postfix}]",
            dynamic_ncols=True, leave=True, position=0,
        )

    def update(self, **postfix) -> None:
        self.bar.set_postfix({k: f"{v:.3f}" for k, v in postfix.items()})
        self.bar.update(1)

    def close(self) -> None:
        self.bar.close()
        print(f"обучение заняло {(time.time() - self.t0) / 60:.1f} мин")


def batch_bar(loader, epoch: int, epochs: int):
    return tqdm(loader, desc=f"эпоха {epoch}/{epochs}", unit="бч", dynamic_ncols=True, leave=False, position=1)
