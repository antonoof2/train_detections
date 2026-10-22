import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from pipeline.utils import autocast


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)

    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def postprocess(logits: torch.Tensor, pred_boxes: torch.Tensor):
    """Как в D-FINE/RT-DETR: sigmoid и top-k по всем парам (запрос, класс).

    Возвращает боксы xyxy в долях [0, 1] (B, K, 4), оценки (B, K) и классы (B, K).
    """
    prob = logits.float().sigmoid()
    _, num_queries, num_classes = prob.shape
    scores, idx = prob.flatten(1).topk(num_queries, dim=1)
    labels = idx % num_classes
    boxes = pred_boxes.float().gather(1, (idx // num_classes).unsqueeze(-1).expand(-1, -1, 4))

    return cxcywh_to_xyxy(boxes).clamp(0, 1), scores, labels


def count_matches(pred: dict, gt: dict, conf: float, iou_thr: float, tp, fp, fn) -> None:
    """Жадное сопоставление предсказаний (оценка >= conf) с GT по классам; копит TP/FP/FN."""
    keep = pred["scores"] >= conf
    order = pred["scores"][keep].argsort(descending=True)
    pboxes, plabels = pred["boxes"][keep][order], pred["labels"][keep][order]

    for c in range(len(tp)):
        p, g = pboxes[plabels == c], gt["boxes"][gt["labels"] == c]
        if len(g) == 0:
            fp[c] += len(p)
            continue
        if len(p) == 0:
            fn[c] += len(g)
            continue

        ious = box_iou(p, g)
        used = torch.zeros(len(g), dtype=torch.bool)
        for i in range(len(p)):           # предсказания уже отсортированы по оценке
            row = ious[i].masked_fill(used, -1.0)
            j = int(row.argmax())
            if row[j] >= iou_thr:
                used[j] = True
                tp[c] += 1
            else:
                fp[c] += 1
        fn[c] += int((~used).sum())


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, names: list[str], conf: float = 0.3, amp: str = "none"
) -> dict[str, float]:
    """Метрики на val: mAP50, mAP50-95, AP по классам, precision/recall/F1 (IoU 0.5, оценка >= conf), val_loss.

    Все боксы считаются в пространстве SxS: масштабирование по осям не меняет IoU, так что метрики
    совпадают с метриками в исходных координатах кадра.
    """
    model.eval()
    size = loader.dataset.size
    scale = torch.tensor([size, size, size, size], dtype=torch.float32)
    num_classes = len(names)
    mean_ap = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    tp, fp, fn = (torch.zeros(num_classes) for _ in range(3))
    loss_sum, batches = 0.0, 0

    for images, targets in loader:
        labels = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast(device, amp):
            out = model(pixel_values=images.to(device, non_blocking=True), labels=labels)
        loss_sum += float(out.loss)
        batches += 1

        boxes, scores, classes = (x.cpu() for x in postprocess(out.logits, out.pred_boxes))
        preds = [{"boxes": b * scale, "scores": s, "labels": c} for b, s, c in zip(boxes, scores, classes)]
        gts = [{"boxes": cxcywh_to_xyxy(t["boxes"]) * scale, "labels": t["class_labels"]} for t in targets]

        mean_ap.update(preds, gts)
        for pred, gt in zip(preds, gts):
            count_matches(pred, gt, conf, 0.5, tp, fp, fn)

    res = mean_ap.compute()
    precision = float(tp.sum() / (tp.sum() + fp.sum()).clamp(min=1))
    recall = float(tp.sum() / (tp.sum() + fn.sum()).clamp(min=1))

    metrics = {
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / max(precision + recall, 1e-9),
        "mAP50": max(float(res["map_50"]), 0.0),         # torchmetrics отдаёт -1, если GT нет вообще
        "mAP50-95": max(float(res["map"]), 0.0),
        "val_loss": loss_sum / max(batches, 1),
    }
    per_class = res["map_per_class"].reshape(-1).tolist()
    for cls, ap in zip(res["classes"].reshape(-1).tolist(), per_class):
        metrics[f"AP/{names[cls]}"] = max(ap, 0.0)

    return metrics
