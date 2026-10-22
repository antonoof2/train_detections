# Fire Detection (RF-DETR)

Детекция людей, огня и дыма на изображениях и видео на базе **RF-DETR**.
Конфигурация — **Hydra**, логирование экспериментов — **ClearML**.

## Классы

| id | класс  | описание |
|:--:|--------|----------|
| 0  | people | Человек  |
| 1  | fire   | Огонь    |
| 2  | smoke  | Дым      |

## Подготовка

> Перед запуском не забудьте активировать окружение:
> ```bash
> conda activate fire
> ```

Установка зависимостей:

```bash
make install
```

## Запуск

**Stage 1** — обучение с COCO-весов:

```bash
make stage1
make stage1 MODEL=medium ARGS="train.epochs=10 train.resolution=640"
```

**Stage 2** — дообучение с весов stage 1:

```bash
make stage2 WEIGHTS=outputs/<дата>/<время>/weights/checkpoint_best_total.pth
```

**Напрямую через Python** — любые Hydra-оверрайды:

```bash
python main.py train.epochs=10 data.train=stage2/images
```

| Переменная | По умолчанию | Описание                                 |
|------------|--------------|------------------------------------------|
| `MODEL`    | `large`      | `nano` \| `small` \| `medium` \| `large` |
| `WEIGHTS`  | —            | путь к `.pth` для stage 2                |
| `ARGS`     | —            | дополнительные Hydra-оверрайды           |

## Конфигурация

Все параметры описаны в [`configs/config.yaml`](configs/config.yaml):

- `data` — путь к датасету и сплиты
- `model` — размер модели и стартовые веса
- `train` — гиперпараметры обучения
- `samples` — фиксированные кадры из val для визуального сравнения моделей
- `predict` — инференс на видео после обучения
