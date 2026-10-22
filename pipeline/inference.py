import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from pipeline.evaluate import postprocess
from pipeline.utils import autocast


class Detector:
    """Предсказание на BGR-кадре: сжатие в квадрат resolution x resolution (как при обучении), деление на 255."""

    def __init__(self, model: nn.Module, size: int, device: torch.device, amp: str = "none"):
        self.model = model.eval()
        self.size = size
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает боксы xyxy в пикселях кадра, оценки и классы для всех предсказаний с оценкой >= conf."""
        h, w = frame_bgr.shape[:2]
        rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        rgb = rgb.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)

        with autocast(self.device, self.amp):
            out = self.model(pixel_values=x)
        boxes, scores, labels = postprocess(out.logits, out.pred_boxes)

        keep = scores[0] >= conf
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
        xyxy = (boxes[0][keep] * scale).cpu().numpy()

        return xyxy, scores[0][keep].cpu().numpy(), labels[0][keep].cpu().numpy()
