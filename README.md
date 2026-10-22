# Fire Detection (D-FINE)

Детекция людей, огня и дыма на изображениях и видео на базе **D-FINE**. Конфигурация — **Hydra**, логирование экспериментов — **ClearML**.


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

## Датасет

YOLO-формат: пути сплитов в `configs/config.yaml` задаются относительно `data.root`,
разметка лежит в соседней папке `labels` (`.../images/x.jpg` → `.../labels/x.txt`).
Кадр без `.txt` или с пустым `.txt` считается кадром без объектов (фон) — такие кадры участвуют в обучении.

```
dataset/
├── stage1/{images,labels}
├── stage2/{images,labels}
├── val/{images,labels}
└── test_video.mp4
```

## Запуск

**Stage 1** — обучение с COCO-весов (скачиваются с HF Hub автоматически):

```bash
make stage1
make stage1 MODEL=medium ARGS="train.epochs=30 train.resolution=800"
```

**Stage 2** — дообучение с весов stage 1:

```bash
make stage2 WEIGHTS=outputs/<дата>/<время>/weights/best
```

**Напрямую через Python** — любые Hydra-оверрайды:

```bash
python main.py train.epochs=10 data.train=stage2/images
```

| Переменная | По умолчанию | Описание                                          |
|------------|--------------|---------------------------------------------------|
| `MODEL`    | `nano`       | `nano` \| `small` \| `medium` \| `large` \| `xlarge` |
| `WEIGHTS`  | —            | папка с чекпоинтом (`.../weights/best`) для stage 2 |
| `ARGS`     | —            | дополнительные Hydra-оверрайды                    |

## Размеры модели

| `model.size` | параметров (≈) | COCO-веса на HF Hub                      |
|--------------|----------------|------------------------------------------|
| `nano`       | 3.8 M          | `ustc-community/dfine-nano-coco`         |
| `small`      | 10.3 M         | `ustc-community/dfine-small-coco`        |
| `medium`     | 19.6 M         | `ustc-community/dfine-medium-coco`       |
| `large`      | 31.3 M         | `ustc-community/dfine-large-coco`        |
| `xlarge`     | 62.8 M         | `ustc-community/dfine-xlarge-coco`       |

Для small/medium/large/xlarge есть и более сильные веса Objects365 → COCO. Они подключаются через `model.weights`:

```bash
python main.py model.size=medium model.weights=ustc-community/dfine-medium-obj2coco
```
