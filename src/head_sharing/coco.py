"""One explicit-input COCO bbox evaluator with standard detection parameters."""
from __future__ import annotations

import argparse
import copy
import json
import math
from numbers import Integral
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def score(annotation: dict, image_ids: list[int], predictions: list[dict]) -> dict:
    def valid_id(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    if not isinstance(image_ids, list) or any(not valid_id(value) for value in image_ids):
        raise ValueError("evaluation image IDs must be integers")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate evaluation image IDs")
    known = {row["id"] for row in annotation.get("images", []) if valid_id(row.get("id"))}
    if not set(image_ids) <= known:
        raise ValueError("evaluation ID missing from supplied annotations")
    categories = {row["id"] for row in annotation.get("categories", []) if valid_id(row.get("id"))}
    for pred in predictions:
        if not isinstance(pred, dict) or not valid_id(pred.get("image_id")) or not valid_id(pred.get("category_id")):
            raise ValueError("prediction image/category IDs must be integers")
        if pred["image_id"] not in image_ids or pred["category_id"] not in categories:
            raise ValueError("prediction image/category outside supplied evaluation set")
        bbox = pred.get("bbox")
        score_value = pred.get("score")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)
                or not isinstance(score_value, (int, float)) or not math.isfinite(score_value)
                or bbox[2] < 0 or bbox[3] < 0):
            raise ValueError("prediction bbox and score must be finite with nonnegative dimensions")
    gt = COCO(); gt.dataset = copy.deepcopy(annotation); gt.createIndex()
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO(); dt.dataset = {"images": [x for x in gt.dataset["images"] if int(x["id"]) in image_ids],
                                   "categories": gt.dataset["categories"], "annotations": []}; dt.createIndex()
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = list(image_ids)
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"]  # T,R,K,A,M; -1 denotes absent
    areas = ev.params.areaRng
    names = ["all", "small", "medium", "large"]
    result = {}
    for label, area_name in (("AP50:95", "all"), ("AP50", "all"), ("AP75", "all"),
                             ("APs", "small"), ("APm", "medium"), ("APl", "large")):
        ai = names.index(area_name); ti = 0 if label == "AP50" else (5 if label == "AP75" else slice(None))
        values = p[ti, :, :, ai, 2]
        valid = values[values > -1]
        result[label] = float(valid.mean()) if valid.size else -1.0
    result["maxDets"] = [1, 10, 100]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score"); p.add_argument("--annotations", required=True); p.add_argument("--ids", required=True); p.add_argument("--predictions", required=True)
    args = parser.parse_args(argv)
    result = score(json.loads(Path(args.annotations).read_text()), json.loads(Path(args.ids).read_text()),
                   json.loads(Path(args.predictions).read_text()))
    print(json.dumps(result, sort_keys=True)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
