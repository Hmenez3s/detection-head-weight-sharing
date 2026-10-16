import numpy as np
import pytest

from head_sharing.geometry import COCO80_TO_91, decoded_rows, inverse_boxes, letterbox, targets_xywhn


def test_odd_letterbox_flip_and_exact_inverse():
    image = np.zeros((101, 203, 3), dtype=np.uint8)
    box = np.array([[13., 17., 100., 85.]])
    for flip in (False, True):
        meta = letterbox(image, 512, flip)
        assert meta.pad[0] != 0 or meta.pad[1] != 0
        target = targets_xywhn(box, np.array([0]), meta)
        xywh = target[:, 1:] * 512
        model_box = np.column_stack((xywh[:,0]-xywh[:,2]/2, xywh[:,1]-xywh[:,3]/2,
                                     xywh[:,0]+xywh[:,2]/2, xywh[:,1]+xywh[:,3]/2))
        assert np.allclose(inverse_boxes(model_box, meta), box, atol=1e-5)
    assert len(targets_xywhn(np.empty((0,4)), np.empty(0), letterbox(image, 640))) == 0
    meta = letterbox(image, 512, flipped=True)
    transformed = targets_xywhn(box, np.array([0]), meta)[:, 1:] * 512
    cx, cy, bw, bh = transformed[0]
    model_row = np.array([[cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2, .9, 0]])
    row = decoded_rows(model_row, 1, meta=meta)[0]
    assert np.allclose(row["bbox"], [13, 17, 87, 68], atol=1e-5)
    with pytest.raises(ValueError): letterbox(image, 513)


def test_coco_category_map_and_native_decoded_rows():
    assert COCO80_TO_91[0] == 1 and COCO80_TO_91[11] == 13 and COCO80_TO_91[-1] == 90
    pred = decoded_rows(np.array([[1,2,11,12,.9,11],[0,0,2,2,.0001,0]]), 9)
    assert len(pred) == 1 and pred[0]["category_id"] == 13 and pred[0]["bbox"] == [1.,2.,10.,10.]
    with pytest.raises(ValueError, match="finite"):
        decoded_rows(np.array([[0,0,1,1,np.nan,0]]), 9)
    with pytest.raises(ValueError, match="nonnegative"):
        decoded_rows(np.array([[2,0,1,1,.9,0]]), 9)
