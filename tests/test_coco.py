from head_sharing.coco import score


def test_genuine_cocoeval_perfect_and_negative_oracles():
    annotation = {"info": {}, "licenses": [], "images": [{"id": 4, "width": 100, "height": 100}],
                 "categories": [{"id": 13, "name": "class"}, {"id": 14, "name": "wrong"}],
                 "annotations": [{"id": 1, "image_id": 4, "category_id": 13,
                                  "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
    perfect = [{"image_id": 4, "category_id": 13, "bbox": [10,10,20,20], "score": .99}]
    metrics = score(annotation, [4], perfect)
    assert all(abs(metrics[k] - 1) < 1e-12 for k in ("AP50:95", "AP50", "AP75"))
    assert metrics["maxDets"] == [1,10,100]
    assert score(annotation, [4], [{**perfect[0], "category_id": 14}])["AP50:95"] < 1
    assert score(annotation, [4], [{**perfect[0], "bbox": [60,60,20,20]}])["AP50:95"] < 1
    empty = score(annotation, [4], [])
    assert empty["AP50:95"] == 0
    assert empty["APm"] == -1 and empty["APl"] == -1


def test_native_signed_dimensions_match_direct_cocoeval():
    import copy
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annotation = {"info": {}, "images": [{"id": 1, "width": 100, "height": 100}],
                  "categories": [{"id": 1, "name": "class"}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                   "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [80,80,-20,-20], "score": .99},
        {"image_id": 1, "category_id": 1, "bbox": [80,80,-20,20], "score": .98},
        {"image_id": 1, "category_id": 1, "bbox": [10,10,20,20], "score": .9},
    ]
    gt = COCO(); gt.dataset = copy.deepcopy(annotation); gt.createIndex()
    native = COCOeval(gt, gt.loadRes(copy.deepcopy(predictions)), "bbox")
    native.params.imgIds = [1]; native.params.maxDets = [1,10,100]
    native.evaluate(); native.accumulate()
    precision = native.eval["precision"][:, :, :, 0, 2]
    expected = float(precision[precision > -1].mean())
    # Two negative dimensions have positive area and can count as a false
    # positive. Dropping inverted rows would incorrectly turn this into AP=1.
    assert 0 < expected < 1
    assert score(annotation, [1], copy.deepcopy(predictions))["AP50:95"] == expected


def test_evaluator_rejects_nonfinite_and_malformed_predictions():
    import pytest
    annotation = {"info": {}, "licenses": [], "images": [{"id": 1, "width": 100, "height": 100}],
                  "categories": [{"id": 1, "name": "class"}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                   "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}]}
    base = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .9}
    for invalid in ({**base, "score": float("nan")},
                    {**base, "bbox": [0, 0, float("inf"), 10]},
                    {**base, "image_id": "1"}, {**base, "category_id": 1.5}):
        with pytest.raises(ValueError):
            score(annotation, [1], [invalid])
