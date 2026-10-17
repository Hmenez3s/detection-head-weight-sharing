"""Fixed letterbox, batch image flip, target conversion and detector decoding."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

COCO80_TO_91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

@dataclass(frozen=True)
class Transform:
    image: np.ndarray
    ratio: tuple[float, float]
    pad: tuple[int, int]
    original_shape: tuple[int, int]
    flipped: bool


def letterbox(image: np.ndarray, imgsz: int = 640, flipped: bool = False) -> Transform:
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32 and at least 32")
    h, w = image.shape[:2]
    pixels = np.ascontiguousarray(image[:, ::-1]) if flipped else image
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True, center=True, stride=32)
    params = lb.get_params({"img": pixels})
    out = lb.apply_image({"img": pixels}, params)["img"]
    return Transform(out, tuple(params["ratio"]), (params["left"], params["top"]), (h, w), flipped)


def targets_xywhn(boxes_xyxy: np.ndarray, classes: np.ndarray, meta: Transform) -> np.ndarray:
    """Return [class,cx,cy,w,h] normalized to square model input."""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    c = np.asarray(classes).reshape(-1, 1)
    if len(b) != len(c):
        raise ValueError("box/class count mismatch")
    h, w = meta.original_shape
    if meta.flipped and len(b):
        x1 = w - b[:, 2].copy(); x2 = w - b[:, 0].copy(); b[:, 0], b[:, 2] = x1, x2
    rx, ry = meta.ratio; px, py = meta.pad
    b[:, [0,2]] = b[:, [0,2]] * rx + px
    b[:, [1,3]] = b[:, [1,3]] * ry + py
    b[:, [0,2]] /= meta.image.shape[1]; b[:, [1,3]] /= meta.image.shape[0]
    xywh = np.column_stack(((b[:,0]+b[:,2])/2,(b[:,1]+b[:,3])/2,b[:,2]-b[:,0],b[:,3]-b[:,1]))
    return np.concatenate((c, xywh), axis=1)


def inverse_boxes(boxes_xyxy, meta: Transform) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1,4).copy()
    # Exact forward gains and integer left/top from LetterBox, including asymmetric odd padding.
    ratio_pad = ((meta.ratio[1], meta.ratio[0]), meta.pad)
    out = scale_boxes(meta.image.shape[:2], boxes, meta.original_shape, ratio_pad=ratio_pad)
    if meta.flipped and len(out):
        w = meta.original_shape[1]
        x1 = w - out[:,2].copy(); x2 = w - out[:,0].copy(); out[:,0], out[:,2] = x1, x2
    return np.asarray(out)


def decoded_rows(rows, image_id: int, category_map=COCO80_TO_91, conf_threshold: float = .001, max_det: int = 300, meta: Transform | None = None) -> list[dict]:
    """Convert native decoded [x1,y1,x2,y2,score,class] rows; no NMS."""
    raw = rows.detach().cpu().numpy() if isinstance(rows, torch.Tensor) else rows
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoded rows must be numeric") from exc
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("decoded rows must have shape (N,6)")
    if not np.isfinite(arr).all():
        raise ValueError("decoded rows must be finite")
    # Native YOLO26 end-to-end export preserves finite inverted endpoints.
    # Keep their signed sizes for COCOeval, rather than swapping or dropping.
    if not np.isfinite(conf_threshold) or max_det < 1:
        raise ValueError("invalid confidence threshold or max_det")
    selected = arr[arr[:,4] >= conf_threshold]
    selected = selected[np.argsort(-selected[:,4], kind="stable")[:max_det]].copy()
    if meta is not None and len(selected):
        selected[:, :4] = inverse_boxes(selected[:, :4], meta)
    result = []
    for x1,y1,x2,y2,score,cls in selected:
        ci = int(cls)
        if ci != cls or ci < 0 or ci >= len(category_map):
            raise ValueError("class index outside category map")
        result.append({"image_id": int(image_id), "category_id": int(category_map[ci]),
                       "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)], "score": float(score)})
    return result
