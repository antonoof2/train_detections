# Fire Detection (DEIMv2)

Детекция людей, огня и дыма на изображениях и видео на базе **[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)**. Конфигурация — **Hydra**, логирование экспериментов — **ClearML**.


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

Установка зависимостей (клонирует официальный код DEIMv2 в `third_party/DEIMv2` на фиксированном коммите и ставит пакеты):

```bash
make install
```

Из репозитория DEIMv2 берутся только модель (бэкбон, энкодер, декодер) и loss; данные, цикл обучения,
метрики и логирование — свои.

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
make stage1 ARGS="model.size=medium train.epochs=30 train.resolution=800"
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
| `WEIGHTS`  | —            | папка с чекпоинтом (`.../weights/best`) для stage 2 |
| `ARGS`     | —            | дополнительные Hydra-оверрайды                    |

## Размеры модели

| `model.size` | бэкбон   | параметров (≈) | AP COCO | COCO-веса на HF Hub                       |
|--------------|----------|----------------|---------|-------------------------------------------|
| `atto`       | HGNetv2  | 0.5 M          | 23.8    | `Intellindust/DEIMv2_HGNetv2_ATTO_COCO`   |
| `femto`      | HGNetv2  | 1.0 M          | 31.0    | `Intellindust/DEIMv2_HGNetv2_FEMTO_COCO`  |
| `pico`       | HGNetv2  | 1.5 M          | 38.5    | `Intellindust/DEIMv2_HGNetv2_PICO_COCO`   |
| `nano`       | HGNetv2  | 3.6 M          | 43.0    | `Intellindust/DEIMv2_HGNetv2_N_COCO`      |
| `small`      | DINOv3   | 9.7 M          | 50.9    | `Intellindust/DEIMv2_DINOv3_S_COCO`       |
| `medium`     | DINOv3   | 18.1 M         | 53.0    | `Intellindust/DEIMv2_DINOv3_M_COCO`       |
| `large`      | DINOv3   | 32.2 M         | 56.0    | `Intellindust/DEIMv2_DINOv3_L_COCO`       |
| `xlarge`     | DINOv3   | 50.3 M         | 57.8    | `Intellindust/DEIMv2_DINOv3_X_COCO`       |

Веса скачиваются с HF Hub автоматически; отдельные веса бэкбона (`ckpts/`) не нужны — грузится вся COCO-модель,
пересоздаются только классификационные головы. Для DINOv3-вариантов ImageNet-нормализация входа делается внутри модели.

Особенности:
- `train.resolution` задаёт и размер, под который строятся якоря для инференса: модель предсказывает именно на нём
  (сохраняется в `config.json` чекпоинта как `eval_spatial_size`).
- `val_loss` не считается — loss DEIMv2 требует aux-выходов декодера, которые есть только в train-режиме.
- Аугментации — как в DEIMv2, но без Mosaic / MixUp / CopyBlend.
