# Fire Detection (D-FINE)

Детекция людей, огня и дыма на изображениях и видео на базе **D-FINE** (реализация из 🤗 `transformers`,
предобученные веса `ustc-community/dfine-*`). Конфигурация — **Hydra**, логирование экспериментов — **ClearML**.

Пайплайн повторяет структуру проекта на RF-DETR: тот же формат датасета (YOLO), те же этапы `stage1`/`stage2`,
те же имена метрик и отчётов в ClearML — модели можно сравнивать между собой.

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

YOLO-формат, как и раньше: пути сплитов в `configs/config.yaml` задаются относительно `data.root`,
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

(`ustc-community/dfine-small-obj2coco`, `...-medium-obj2coco`, `...-large-obj2coco-e25`, `...-xlarge-obj2coco`;
для nano есть только COCO.) При загрузке классификационные головы (80 классов COCO) пересоздаются под ваши 3 класса,
остальные веса остаются предобученными.

## Конфигурация

Все параметры описаны в [`configs/config.yaml`](configs/config.yaml):

- `data` — путь к датасету, сплиты, имена классов
- `model` — размер модели и стартовые веса (`null` — COCO; id с HF Hub или папка — дообучение)
- `train` — гиперпараметры обучения
- `samples` — фиксированные кадры из val для визуального сравнения моделей
- `predict` — инференс на видео после обучения

Что важно знать про `train`:

- `resolution` кратно 32, кадры сжимаются в квадрат `resolution × resolution` (на инференсе — так же).
- Эффективный батч = `batch_size × grad_accum_steps`. Не хватает памяти — уменьшайте `batch_size`,
  увеличивайте `grad_accum_steps`, при необходимости выключайте `multiscale`.
- `lr` — для энкодера, декодера и голов, `lr_backbone` — для бэкбона HGNetV2 (у него lr меньше).
- `ema: true` — метрики, `best`/`last` и инференс считаются по EMA-весам (как в оригинальном D-FINE).
- `multiscale` — каждый батч обучается на случайном размере из `[resolution − range, resolution + range]`.
- `no_aug_last_epochs` — в последних эпохах выключаются сильные аугментации и multiscale (так делает оригинал).
- `amp: bf16 | fp16 | none` — на GPU без поддержки bf16 автоматически включится fp16.
- Выбор `best` — по mAP50-95 на val, по EMA-весам; ранняя остановка — по нему же.

Эпох по умолчанию 5, как в проекте на RF-DETR (для сравнения «в лоб»). D-FINE обычно выигрывает от более
длинного обучения — для финальной модели ставьте 20–50 эпох.

## Что получается в итоге

В `outputs/<дата>/<время>/`:

- `weights/best` — лучший чекпоинт по mAP50-95 (папка в формате `save_pretrained`: `config.json` с классами
  и `model.safetensors`), `weights/last` — после последней эпохи;
- `<имя видео>_pred.mp4` — видео с предсказаниями.

В ClearML:

- скаляры по эпохам: `loss/train`, `loss_parts/*` (vfl, bbox, giou, fgl, ddf), `lr/*`, `val/*`
  (mAP50, mAP50-95, precision, recall, F1, val_loss);
- `val/precision`, `val/recall`, `val/mAP50`, `val/mAP50-95`, `val/F1` и таблица AP по классам — для лучших весов
  (те же имена, что в проекте на RF-DETR);
- Debug Samples `samples` — предсказания на фиксированных кадрах из `samples.images`;
- артефакты `best_weights` и `predicted_video`.

Precision/recall/F1 считаются при IoU 0.5 и пороге уверенности `predict.conf`.

## Работа без интернета на сервере обучения

Один раз скачайте веса там, где интернет есть, и положите на сервер:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ustc-community/dfine-nano-coco', local_dir='weights/dfine-nano-coco')"
```

Дальше — `model.weights=weights/dfine-nano-coco` и `HF_HUB_OFFLINE=1`.

## Структура проекта

```
main.py              # точка входа: обучение → валидация → сэмплы → видео
configs/config.yaml
pipeline/
├── data.py          # YOLO-датасет, аугментации, загрузчики
├── model.py         # загрузка модели, EMA, оптимизатор, расписание lr
├── train.py         # цикл обучения, best/last, ранняя остановка
├── evaluate.py      # постобработка, mAP и precision/recall/F1
├── inference.py     # предсказание на одном кадре
├── progress.py      # полосы прогресса по эпохам и батчам
└── report.py        # отчёты в ClearML: метрики, сэмплы, видео
```
