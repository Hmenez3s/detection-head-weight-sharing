import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from head_sharing.checkpoint import load_checkpoint, load_resume, save_checkpoint, save_resume
from head_sharing.head_spec import canonicalize_head_spec, head_spec_digest
from head_sharing.model import HeadStudyModel
from head_sharing.train import _native_criterion, optimizer_groups

RECEIPT = {
    "trunk_artifact_sha256": "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016",
    "trunk_tensors": 468, "trunk_bytes": 9_109_168, "head_index": 23,
    "input_channels": [64, 128, 256], "source_variant": "U", "source_seed": 11,
    "source_epoch_zero_based": 79, "source_training_epochs": 80,
    "pre_exposure": "Synthetic test receipt.",
}
VARIANTS = ("spatial_untied", "spatial_sepbn", "spatial_gn", "spatial_sepbn_k5", "spatial_sepbn_sharedpred")


def spec(variant):
    return {"version": 2, "variant": variant}


@pytest.mark.parametrize("variant", VARIANTS)
def test_native_spatial_variants_train_roundtrip_and_preserve_semantics(tmp_path, variant):
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    model = HeadStudyModel("native", 11, 32, head_spec=spec(variant))
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.head_spec == spec(variant)

    native = HeadStudyModel("native", 11, 32)
    spatial_weight_ids = []
    spatial_norm_ids = []
    features = [torch.randn(2, c, n, n) for c, n in ((64, 4), (128, 2), (256, 1))]
    if variant == "spatial_untied":
        matched_shared = HeadStudyModel("native", 11, 32, head_spec=spec("spatial_sepbn")).eval()
        model.eval()
        output_untied = model.forward_features(features)
        output_shared = matched_shared.forward_features(features)
        for branch in ("one2many", "one2one"):
            for key in ("boxes", "scores"):
                assert torch.equal(output_untied[1][branch][key], output_shared[1][branch][key])
    for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branch = "box" if "cv2" in name else "class"
        modules = getattr(model.detect, name)
        original = getattr(native.detect, name)
        sites = ((1,),) if branch == "box" else ((1, 0), (1, 1))
        for site in sites:
            blocks = [modules[level][site[0]] if len(site) == 1 else modules[level][site[0]][site[1]] for level in range(3)]
            originals = [original[level][site[0]] if len(site) == 1 else original[level][site[0]][site[1]] for level in range(3)]
            if variant in ("spatial_untied", "spatial_sepbn"):
                assert torch.equal(blocks[0].conv.weight, blocks[1].conv.weight)
            if variant == "spatial_untied":
                assert len({id(block.conv.weight) for block in blocks}) == 3
            else:
                assert blocks[0].conv.weight is blocks[1].conv.weight is blocks[2].conv.weight
            spatial_weight_ids.append(id(blocks[0].conv.weight))
            if variant != "spatial_untied":
                assert blocks[0].conv.weight is blocks[1].conv.weight
            if variant == "spatial_gn":
                assert blocks[0].bn is blocks[1].bn is blocks[2].bn
                assert isinstance(blocks[0].bn, nn.GroupNorm)
                assert blocks[0].bn.num_groups == (8 if branch == "box" else 16)
                spatial_norm_ids.append(id(blocks[0].bn))
            else:
                assert all(isinstance(block.bn, nn.BatchNorm2d) for block in blocks)
                assert len({id(block.bn) for block in blocks}) == 3
                assert len({id(block.bn.weight) for block in blocks}) == 3
                assert len({id(block.bn.bias) for block in blocks}) == 3
                for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                    assert len({block.bn.get_buffer(buffer_name).data_ptr() for block in blocks}) == 3
            if variant == "spatial_sepbn_k5" and originals[0].conv.kernel_size == (3, 3):
                assert [block.conv.kernel_size for block in blocks] == [(5, 5)] * 3
                assert torch.equal(blocks[0].conv.weight[:, :, 1:4, 1:4], originals[0].conv.weight)
                assert torch.count_nonzero(blocks[0].conv.weight[:, :, 0, :]) == 0
            elif not (branch == "class" and site == (1, 0) and variant == "spatial_sepbn_k5"):
                assert blocks[0].conv.kernel_size == originals[0].conv.kernel_size
        # The first block at every scale remains native and untouched.
        assert all(torch.equal(value, original[level][0].state_dict()[key])
                   for level in range(3) for key, value in modules[level][0].state_dict().items())
    assert len(spatial_weight_ids) == len(set(spatial_weight_ids))
    if variant == "spatial_gn":
        assert len(spatial_norm_ids) == len(set(spatial_norm_ids)) == 6
    for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        modules = getattr(model.detect, name)
        assert len({id(modules[level][-1].bias) for level in range(3)}) == 3
    if variant == "spatial_sepbn_sharedpred":
        for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            modules = getattr(model.detect, name)
            assert modules[0][-1].weight is modules[1][-1].weight is modules[2][-1].weight
            assert len({id(modules[i][-1].bias) for i in range(3)}) == 3

    expected_counts = {"spatial_untied": 309656, "spatial_sepbn": 271960,
                       "spatial_gn": 270552, "spatial_sepbn_k5": 282712,
                       "spatial_sepbn_sharedpred": 246104}
    unique = {id(p): p for p in model.detect.parameters()}
    assert sum(p.numel() for p in unique.values()) == expected_counts[variant]
    trunk_before = {name: value.clone() for name, value in model.detector.model[:-1].state_dict().items()}
    features = [torch.randn(2, c, n, n) for c, n in ((64, 4), (128, 2), (256, 1))]
    if variant == "spatial_sepbn":
        from head_sharing.profile import measure_conv_macs
        native_macs = measure_conv_macs(native.detect, lambda: native.forward_features(features))
        shared_macs = measure_conv_macs(model.detect, lambda: model.forward_features(features))
        assert native_macs["conv_macs"] == shared_macs["conv_macs"]
        assert native_macs["conv_invocations"] == shared_macs["conv_invocations"]
    model.train()
    prediction = model.forward_features(features)
    criterion = _native_criterion(model, 2)
    targets = {"batch_idx": torch.tensor([0.]), "cls": torch.tensor([[1.]]),
               "bboxes": torch.tensor([[.5, .5, .2, .2]])}
    loss, _ = criterion(prediction, targets)
    assert torch.isfinite(loss).all() and loss.sum().item() > 0
    optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01, momentum=.9)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    no_decay = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert all(id(p) in no_decay for m in model.detect.modules() if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) for p in m.parameters(recurse=False))
    optimizer.zero_grad(set_to_none=True); loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    before = next(p for p in params if p.grad is not None).detach().clone()
    changed = next(p for p in params if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, changed)
    assert all(torch.equal(value, model.detector.model[:-1].state_dict()[name]) for name, value in trunk_before.items())

    model.eval()
    expected = model.forward_features(features)
    import copy
    pruned = copy.deepcopy(model)
    pruned.detect.fuse()
    deployed = pruned.forward_features(features)
    assert pruned.detect.cv2 is None and pruned.detect.cv3 is None
    assert torch.equal(expected[0], deployed[0])
    for key in ("boxes", "scores"):
        assert torch.equal(expected[1]["one2one"][key], deployed[1]["one2one"][key])
    from head_sharing.profile import measure_conv_macs
    macs = measure_conv_macs(model.detect, lambda: model.forward_features(features))
    assert macs["conv_macs"] > 0 and macs["conv_invocations"] > 0
    trunk = {k: v.detach().clone() for k, v in model.detector.state_dict().items()
             if k.startswith("model.") and int(k.split(".")[1]) < 23}
    artifact = {"state_dict": trunk, "schema_version": 1, "architecture": "yolo26n",
                "upstream": "8.4.177", "excluded_head_index": 23}
    weights, config = tmp_path / "head.pt", tmp_path / "head.json"
    save_checkpoint(model, weights, config, RECEIPT)
    assert json.loads(config.read_text())["format"] == 2
    restored = load_checkpoint(weights, config, trunk_artifact=artifact, synthetic_trunk=True)
    actual = restored.forward_features(features)
    for branch in ("one2many", "one2one"):
        for key in ("boxes", "scores"):
            assert torch.equal(expected[1][branch][key], actual[1][branch][key])
    assert restored.head_spec == spec(variant)
    assert head_spec_digest(restored.head_spec) == head_spec_digest(spec(variant))

    resume = tmp_path / "resume.pt"
    save_resume(model, optimizer, resume, epoch=1)
    wrong = HeadStudyModel("native", 11, 32, head_spec=spec("spatial_untied" if variant != "spatial_untied" else "spatial_sepbn"))
    wrong_opt = torch.optim.SGD(optimizer_groups(wrong, .0005), lr=.01, momentum=.9)
    model_before = {k: v.clone() for k, v in wrong.detect.state_dict().items()}
    optimizer_before = copy.deepcopy(wrong_opt.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="identity"):
        load_resume(wrong, wrong_opt, resume)
    assert all(torch.equal(v, wrong.detect.state_dict()[k]) for k, v in model_before.items())
    assert wrong_opt.state_dict() == optimizer_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    if variant != "spatial_untied":
        from head_sharing.checkpoint import _alias_groups
        payload = torch.load(resume, weights_only=True)
        groups = [[name.removeprefix("detector.model.23.") for name in group
                   if name.startswith("detector.model.23.")]
                  for group in _alias_groups(model)]
        aliases = next(group for group in groups if len(group) > 1 and all(name in payload["model"] for name in group))
        payload["model"][aliases[1]] = payload["model"][aliases[1]].clone() + 1
        conflicting = tmp_path / "resume-alias-conflict.pt"
        torch.save(payload, conflicting)
        valid_target = HeadStudyModel("native", 11, 32, head_spec=spec(variant))
        valid_optimizer = torch.optim.SGD(optimizer_groups(valid_target, .0005), lr=.01, momentum=.9)
        valid_model_before = {k: v.clone() for k, v in valid_target.detect.state_dict().items()}
        valid_optimizer_before = copy.deepcopy(valid_optimizer.state_dict())
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match="conflicting duplicate"):
            load_resume(valid_target, valid_optimizer, conflicting)
        assert all(torch.equal(v, valid_target.detect.state_dict()[k]) for k, v in valid_model_before.items())
        assert valid_optimizer.state_dict() == valid_optimizer_before
        assert random.getstate() == python_rng_before
        numpy_rng_after = np.random.get_state()
        assert numpy_rng_after[0] == numpy_rng_before[0]
        assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
        assert numpy_rng_after[2:] == numpy_rng_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_v2_schema_is_strict_and_v1_identity_is_unchanged():
    assert canonicalize_head_spec(None) is None
    old = {"version": 1, "variant": "scale_gain", "class_inner": 96}
    assert canonicalize_head_spec(old) == old
    assert head_spec_digest(old) == "9096e0dc882a29210c966813bc52146e8d823ee124801084ff248de8be1d34f5"
    for bad in ({"version": True, "variant": "spatial_gn"}, {"version": 2, "variant": "spatial_gn", "class_inner": 96},
                {"version": 2, "variant": "native"}, {"version": 2, "variant": 3}):
        with pytest.raises(ValueError): canonicalize_head_spec(bad)
    with pytest.raises(ValueError): HeadStudyModel("loop_ab2", 11, 32, head_spec=spec("spatial_gn"))
