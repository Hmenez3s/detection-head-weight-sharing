import copy
import hashlib
import json

import pytest
import torch

from head_sharing.checkpoint import (_read_verified_trunk_artifact, load_checkpoint, load_resume,
                                     load_trunk_artifact, save_checkpoint, save_resume)
from head_sharing.model import HeadStudyModel, main, validate_imgsz


RECEIPT = {
    "trunk_artifact_sha256": "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016",
    "trunk_tensors": 468, "trunk_bytes": 9_109_168, "head_index": 23,
    "input_channels": [64, 128, 256], "source_variant": "U", "source_seed": 11,
    "source_epoch_zero_based": 79, "source_training_epochs": 80,
    "pre_exposure": "Synthetic test receipt: source trunk pre-exposure disclosed.",
}


def test_default_ab2_state_matches_independent_oracle():
    import hashlib
    baseline = HeadStudyModel("loop_ab2", 11, 32)
    explicit = HeadStudyModel("loop_ab2", 11, 32, head_spec={"version": 1, "variant": "baseline", "class_inner": 96})
    digest = hashlib.sha256()
    for name, tensor in baseline.detect.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    assert digest.hexdigest() == "74030da2b78acadc3cdf462191c18d947c9e5d9686b21f50bb62a8ea316ff5ef"
    assert list(baseline.detect.state_dict()) == list(explicit.detect.state_dict())
    assert all(torch.equal(v, explicit.detect.state_dict()[k]) for k, v in baseline.detect.state_dict().items())


def test_default_ab2_outputs_match_prechange_four_tower_oracle():
    import hashlib
    import torch
    from head_sharing.model import HeadStudyModel
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        model = HeadStudyModel("loop_ab2", 11, 32).eval()
        generator = torch.Generator().manual_seed(2026)
        features = [torch.randn(2, channels, side, side, generator=generator).to(torch.float16).float()
                    for channels, side in ((64, 4), (128, 2), (256, 1))]
        digest = hashlib.sha256()
        with torch.inference_mode():
            for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
                tensor = getattr(model.detect, name)[0](features, "loop_ab2")
                digest.update(name.encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        assert digest.hexdigest() == "d015edc3d0de71d45cead22a8bf0ed7a8db8de329494cf8eb8f6335ec3ab75ce"
    finally:
        torch.set_num_threads(previous_threads)


def test_head_spec_canonicalization_alpha_and_diagnostics():
    from head_sharing.head_spec import canonicalize_head_spec, head_spec_digest
    from head_sharing.model import diagnose_ab2_updates
    baseline = {"version": 1, "variant": "baseline", "class_inner": 96}
    alpha_spec = {"version": 1, "variant": "learned_alpha", "class_inner": 96}
    assert canonicalize_head_spec(baseline) is None
    assert head_spec_digest(alpha_spec) == head_spec_digest(dict(alpha_spec))
    for invalid in ({**alpha_spec, "extra": 1}, {"version": True, "variant": "learned_alpha", "class_inner": 96},
                    {"version": 1, "variant": "learned_alpha", "class_inner": True}):
        with pytest.raises(ValueError): canonicalize_head_spec(invalid)
    with pytest.raises(ValueError, match="only for loop_ab2"):
        HeadStudyModel("native", 11, 32, head_spec=alpha_spec)
    for unsupported in ("step_norm", "scale_gain", "dw5"):
        with pytest.raises(ValueError, match="not implemented"):
            HeadStudyModel("loop_ab2", 11, 32, head_spec={"version": 1, "variant": unsupported, "class_inner": 96})

    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    baseline_model = HeadStudyModel("loop_ab2", 11, 32).eval()
    model = HeadStudyModel("loop_ab2", 11, 32, head_spec=alpha_spec).eval()
    assert torch.equal(torch.get_rng_state(), rng_before)
    towers = [model.detect.cv2[0], model.detect.cv3[0], model.detect.one2one_cv2[0], model.detect.one2one_cv3[0]]
    assert all(t.alpha.shape == (2,) and torch.equal(t.alpha, torch.full((2,), .1)) for t in towers)
    features = [torch.randn(2, c, n, n) for c, n in ((64, 4), (128, 2), (256, 1))]
    base_out = baseline_model.forward_features(features)
    alpha_out = model.forward_features(features)
    for key in ("one2many", "one2one"):
        assert torch.equal(base_out[1][key]["boxes"], alpha_out[1][key]["boxes"])
        assert torch.equal(base_out[1][key]["scores"], alpha_out[1][key]["scores"])
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    before_mode = model.training
    before_rng = torch.get_rng_state().clone()
    diagnostic = diagnose_ab2_updates(model, features)
    assert set(diagnostic) == {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}
    assert all(len(records) == 2 and all(torch.isfinite(torch.tensor(list(r.values()))).all() for r in records)
               for records in diagnostic.values())
    assert model.training == before_mode and torch.equal(torch.get_rng_state(), before_rng)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before_state.items())
    from head_sharing.train import optimizer_groups
    optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01)
    optimizer.zero_grad(set_to_none=True)
    sum(output["boxes"].sum() + output["scores"].sum()
        for output in (alpha_out[1]["one2many"], alpha_out[1]["one2one"])).backward()
    assert all(t.alpha.grad is not None and torch.isfinite(t.alpha.grad).all() for t in towers)
    assert all(p.grad is None for p in model.detector.model[:-1].parameters())
    assert all(id(t.alpha) in {id(p) for p in optimizer.param_groups[1]["params"]} for t in towers)
    old_alpha = [tower.alpha.detach().clone() for tower in towers]
    optimizer.step()
    assert all(not torch.equal(old, tower.alpha) for old, tower in zip(old_alpha, towers, strict=True))


def test_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--inspect" in capsys.readouterr().out


def test_facts_intervention_alias_gradients_and_frozen_trunk():
    native = HeadStudyModel("native", seed=11, imgsz=512)
    independent = HeadStudyModel("independent_k2", seed=11, imgsz=512)
    shared = HeadStudyModel("shared_k2", seed=11, imgsz=512)
    assert validate_imgsz(512) == 512
    assert validate_imgsz(640) == 640
    with pytest.raises(ValueError):
        validate_imgsz(513)
    assert [64, 128, 256] == [m[0].conv.in_channels for m in native.detect.cv2]
    assert native.detect.nc == 80 and native.detect.reg_max == 1
    assert native.detect.stride.tolist() == [8, 16, 32]
    assert sum(p.numel() for p in native.detect.parameters()) == 309656

    for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        a, b, c = (getattr(m.detect, attr) for m in (native, independent, shared))
        for site in range(3):
            assert isinstance(a[site][1], torch.nn.Module)
            assert len(b[site][1]) == len(c[site][1]) == 2
            assert b[site][1][0] is not b[site][1][1]
            assert c[site][1][0] is c[site][1][1]
            assert all(torch.equal(x, y) for x, y in zip(
                a[site][1].state_dict().values(), b[site][1][0].state_dict().values(), strict=True
            ))
            for (name_x, value_x), (name_y, value_y) in zip(
                b[site][1][0].state_dict().items(), b[site][1][1].state_dict().items(), strict=True
            ):
                assert value_x.data_ptr() != value_y.data_ptr()
                assert torch.equal(value_x, value_y)
            assert sum(p.numel() for p in b[site][1][0].parameters()) == sum(
                p.numel() for p in c[site][1][0].parameters()
            )

    def unique_head_parameters(model):
        parameters = {id(p): p for p in model.detect.parameters()}
        return sum(p.numel() for p in parameters.values())

    assert unique_head_parameters(shared) == unique_head_parameters(native) == 309656
    trunk_before = {k: v.detach().clone() for k, v in shared.detector.model[:-1].state_dict().items()}
    assert all(not p.requires_grad for p in shared.detector.model[:-1].parameters())
    shared.train()
    assert all(not module.training for layer in shared.detector.model[:-1] for module in layer.modules())
    assert shared.detect.training
    features = [torch.randn(2, c, side, side) for c, side in ((64, 8), (128, 4), (256, 2))]
    output = shared.forward_features(features)
    independent.forward_features(features)
    for branch, independent_branch in ((shared.detect.cv2, independent.detect.cv2),
                                       (shared.detect.one2one_cv2, independent.detect.one2one_cv2)):
        assert branch[0][1][0].bn.num_batches_tracked.item() == 2
        assert independent_branch[0][1][0].bn.num_batches_tracked.item() == 1
        assert independent_branch[0][1][1].bn.num_batches_tracked.item() == 1
    loss = output["one2many"]["boxes"].sum() + output["one2many"]["scores"].sum()
    loss.backward()
    aliased = shared.detect.cv2[0][1][0].conv.weight
    assert aliased.grad is not None
    opt = torch.optim.SGD({id(p): p for p in shared.parameters() if p.requires_grad}.values(), lr=0.01)
    old = aliased.detach().clone()
    opt.step()
    assert not torch.equal(old, aliased)
    assert all(torch.equal(v, trunk_before[k]) for k, v in shared.detector.model[:-1].state_dict().items())
    assert all(p.grad is None for p in shared.detector.model[:-1].parameters())


@pytest.mark.parametrize("arm,widths,classes,passes", [("loop_ab2", 24, 96, 2), ("loop_a3", 28, 108, 3)])
def test_looped_shared_list_native_loss_update_and_fuse(arm, widths, classes, passes):
    from head_sharing.train import _native_criterion, optimizer_groups
    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    model = HeadStudyModel(arm, seed=11, imgsz=64)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.detect.cv2[0].adapters[0][1].out_channels == widths
    assert model.detect.cv3[0].adapters[0][1].out_channels == classes
    assert model.detect.one2one_cv2[0] is not model.detect.cv2[0]
    assert model.detect.cv2[0].a is not model.detect.cv3[0].a
    assert torch.stack(list(model.detect.cv2[0].scale_id)).shape == (3, widths)
    assert torch.count_nonzero(torch.stack(list(model.detect.cv2[0].scale_id))) == 0
    assert model.detect.cv2[0].norm_a.eps == 1e-5
    assert model.detect.cv2[0].a.kernel_size == (1, 1)
    if arm == "loop_ab2":
        assert model.detect.cv2[0].a is not model.detect.cv2[0].b
    assert len([m for m in model.detect.cv2[0].modules() if isinstance(m, torch.nn.Conv2d)]) == 7 + (2 if arm == "loop_ab2" else 1)
    box_tower, cls_tower = model.detect.cv2[0], model.detect.cv3[0]
    assert torch.stack(list(box_tower.scale_bias))[:, 0].tolist() == [2.0] * 3
    expected_class_prior = torch.tensor([
        __import__("math").log(5 / 80 / (640 / stride) ** 2) for stride in (8, 16, 32)])
    assert torch.allclose(torch.stack(list(cls_tower.scale_bias))[:, 0], expected_class_prior)
    assert torch.count_nonzero(box_tower.readout.bias) == torch.count_nonzero(cls_tower.readout.bias) == 0
    assert torch.equal(box_tower.readout.bias[0] + torch.stack(list(box_tower.scale_bias))[:, 0],
                       torch.full((3,), 2.0))
    assert torch.equal(cls_tower.readout.bias[0] + torch.stack(list(cls_tower.scale_bias))[:, 0], expected_class_prior)
    decay, no_decay = optimizer_groups(model, .0005)
    no_decay_ids = {id(p) for p in no_decay["params"]}
    scale_parameters = [p for name, p in model.detect.named_parameters()
                        if "scale_id" in name or "scale_bias" in name]
    assert len(scale_parameters) == 24 and all(p.ndim == 1 and id(p) in no_decay_ids for p in scale_parameters)
    assert not ({id(p) for p in scale_parameters} & {id(p) for p in decay["params"]})
    tower_features = [torch.randn(2, c, n, n) for c, n in ((64, 8), (128, 4), (256, 2))]
    joint = box_tower(tower_features, arm)
    by_level = torch.cat([box_tower([feature], arm, level_indices=[index])
                          for index, feature in enumerate(tower_features)], dim=-1)
    assert torch.allclose(joint, by_level, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        box_tower.readout.bias.fill_(3.0)
        cls_tower.readout.bias.fill_(3.0)
        for bias in box_tower.scale_bias: bias.fill_(-7.0)
        for bias in cls_tower.scale_bias: bias.fill_(-7.0)
    model.detect.bias_init()
    assert torch.count_nonzero(box_tower.readout.bias) == torch.count_nonzero(cls_tower.readout.bias) == 0
    assert torch.stack(list(box_tower.scale_bias))[:, 0].tolist() == [2.0] * 3
    assert torch.allclose(torch.stack(list(cls_tower.scale_bias))[:, 0], expected_class_prior)
    trunk = {k: v.clone() for k, v in model.detector.model[:-1].state_dict().items()}
    features = [torch.randn(2, c, n, n) for c, n in ((64, 8), (128, 4), (256, 2))]
    model.train()
    prediction = model.forward_features(features)
    assert prediction["one2many"]["boxes"].shape == (2, 4, 84)
    criterion = _native_criterion(model, 2)
    targets = {"batch_idx": torch.tensor([0.]), "cls": torch.tensor([[1.]]),
               "bboxes": torch.tensor([[.5, .5, .2, .2]])}
    loss, _ = criterion(prediction, targets)
    assert torch.isfinite(loss).all() and loss.sum().item() > 0
    optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01, momentum=.9)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    head = model.detect.cv2[0].a.weight
    before = head.detach().clone()
    optimizer.zero_grad(set_to_none=True); loss.sum().backward(); optimizer.step()
    assert head.grad is not None and torch.isfinite(head.grad).all() and not torch.equal(before, head)
    assert all(torch.equal(v, trunk[k]) for k, v in model.detector.model[:-1].state_dict().items())
    model.eval()
    unfused = model.forward_features(features)
    from head_sharing.profile import measure_conv_macs
    arithmetic = measure_conv_macs(model.detect, lambda: model.forward_features(features))
    assert arithmetic["conv_invocations"] == (44 if arm == "loop_ab2" else 40)
    assert arithmetic["conv_macs"] > 0
    decoded = unfused[0].clone()
    model.detect.fuse()
    fused = model.forward_features(features)
    assert torch.equal(decoded, fused[0])
    assert torch.equal(unfused[1]["one2one"]["boxes"], fused[1]["one2one"]["boxes"])
    assert torch.equal(unfused[1]["one2one"]["scores"], fused[1]["one2one"]["scores"])
    assert model.detect.cv2 is None
    with pytest.raises(RuntimeError, match="inference-only"):
        model.train()


def test_q_reference_online_features_and_native_decode():
    model = HeadStudyModel("native", seed=22, imgsz=640).eval()
    images = torch.randn(1, 3, 640, 640)
    quantized_features = model.extract_features(images)
    expected = model.detect(quantized_features)
    actual = model.forward_images(images)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1]["one2one"]["boxes"], expected[1]["one2one"]["boxes"])
    persisted_features = [f.to(torch.float16) for f in quantized_features]
    via_cache = model.forward_features(persisted_features)
    assert torch.equal(via_cache[0], actual[0])
    assert torch.equal(via_cache[1]["one2many"]["scores"], actual[1]["one2many"]["scores"])
    assert actual[0].shape == (1, 300, 6)
    assert torch.equal(model.deployment()(images)[0], actual[0])


def test_sanitized_pre_head_container_loads_trunk_without_replacing_fresh_head():
    model = HeadStudyModel("native", seed=44, imgsz=512)
    before_head = {k: v.clone() for k, v in model.detect.state_dict().items()}
    trunk = {k: v.detach().clone() for k, v in model.detector.state_dict().items()
             if k.startswith("model.") and int(k.split(".")[1]) < 23}
    artifact = {"state_dict": trunk, "schema_version": 1, "architecture": "yolo26n",
                "upstream": "8.4.177", "excluded_head_index": 23}
    with pytest.raises(ValueError, match="synthetic test fixtures"):
        load_trunk_artifact(model, artifact, receipt=RECEIPT)
    load_trunk_artifact(model, artifact, receipt=RECEIPT, synthetic=True)
    assert all(torch.equal(model.detect.state_dict()[k], v) for k, v in before_head.items())
    assert all(not p.requires_grad for p in model.detector.model[:-1].parameters())
    corrupted = dict(artifact, state_dict={**trunk, "model.23.cv2.0.0.conv.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="topology"):
        load_trunk_artifact(model, corrupted, receipt=RECEIPT, synthetic=True)
    with pytest.raises(ValueError, match="metadata"):
        load_trunk_artifact(model, dict(artifact, upstream="other"), receipt=RECEIPT, synthetic=True)


def test_verified_trunk_file_hash_rejects_same_shape_mutation(tmp_path):
    model = HeadStudyModel("native", seed=66, imgsz=512)
    state = {k: v.detach().clone() for k, v in model.detector.state_dict().items()
             if k.startswith("model.") and int(k.split(".")[1]) < 23}
    artifact = {"state_dict": state, "schema_version": 1, "architecture": "yolo26n",
                "upstream": "8.4.177", "excluded_head_index": 23}
    path = tmp_path / "synthetic-trunk.pt"
    torch.save(artifact, path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _read_verified_trunk_artifact(path, expected_hash)["excluded_head_index"] == 23
    first_name = next(iter(state))
    changed = state.copy()
    changed[first_name] = state[first_name] + 1
    torch.save(dict(artifact, state_dict=changed), path)
    with pytest.raises(ValueError, match="SHA256"):
        _read_verified_trunk_artifact(path, expected_hash)


def test_cpu_savers_preserve_physical_aliases_when_cpu_copies_clone(tmp_path, monkeypatch):
    model = HeadStudyModel("shared_k2", seed=77, imgsz=512)
    optimizer = torch.optim.SGD({id(p): p for p in model.parameters() if p.requires_grad}.values(), lr=0.01)
    parameter = model.detect.cv2[0][1][0].conv.weight
    before = parameter.detach().clone()
    original_data_ptr = parameter.data_ptr()
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *args, **kwargs: self.clone())
    weights, config, resume = tmp_path / "weights.pt", tmp_path / "config.json", tmp_path / "resume.pt"
    save_checkpoint(model, weights, config, RECEIPT)
    head = torch.load(weights, weights_only=True)
    aliases = [key for key in head if key.startswith("cv2.0.1.") and key.endswith("conv.weight")]
    assert len(aliases) == 2 and head[aliases[0]].untyped_storage().data_ptr() == head[aliases[1]].untyped_storage().data_ptr()
    save_resume(model, optimizer, resume, epoch=1)
    resumed_head = torch.load(resume, weights_only=True)["model"]
    assert resumed_head[aliases[0]].untyped_storage().data_ptr() == resumed_head[aliases[1]].untyped_storage().data_ptr()
    assert parameter.data_ptr() == original_data_ptr and torch.equal(parameter, before)
    assert any(parameter is p for group in optimizer.param_groups for p in group["params"])


def test_resume_rejects_bad_dtype_or_epoch_without_mutating_head_or_optimizer(tmp_path):
    model = HeadStudyModel("shared_k2", seed=88, imgsz=512)
    parameter = next(model.detect.parameters())
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    valid = tmp_path / "resume-valid.pt"
    save_resume(model, optimizer, valid, epoch=2)
    valid_payload = torch.load(valid, weights_only=True)

    for label, mutate in (
        ("dtype", lambda payload: payload["model"].__setitem__(
            next(k for k, v in payload["model"].items() if v.is_floating_point()),
            next(v for v in payload["model"].values() if v.is_floating_point()).half())),
        ("epoch", lambda payload: payload.__setitem__("epoch", -1)),
    ):
        payload = copy.deepcopy(valid_payload)
        mutate(payload)
        bad = tmp_path / f"resume-{label}.pt"
        torch.save(payload, bad)
        before_head = {k: v.detach().clone() for k, v in model.detect.state_dict().items()}
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        with pytest.raises(ValueError, match="invalid resume"):
            load_resume(model, optimizer, bad)
        assert all(torch.equal(v, before_head[k]) for k, v in model.detect.state_dict().items())
        after_optimizer = optimizer.state_dict()
        assert after_optimizer["param_groups"] == before_optimizer["param_groups"]
        assert after_optimizer["state"].keys() == before_optimizer["state"].keys()
        assert all(torch.equal(after_optimizer["state"][pid][key], value)
                   for pid, entry in before_optimizer["state"].items() for key, value in entry.items())


def test_checked_tensor_disk_roundtrip_and_duplicate_alias_rejection(tmp_path):
    model = HeadStudyModel("shared_k2", seed=33, imgsz=512).eval()
    weights, config = tmp_path / "head.pt", tmp_path / "config.json"
    sample = [torch.randn(1, c, n, n) for c, n in ((64, 8), (128, 4), (256, 2))]
    expected = model.forward_features(sample)
    trunk = {k: v.detach().clone() for k, v in model.detector.state_dict().items()
             if k.startswith("model.") and int(k.split(".")[1]) < 23}
    trunk_artifact = {"state_dict": trunk, "schema_version": 1, "architecture": "yolo26n",
                      "upstream": "8.4.177", "excluded_head_index": 23}
    save_checkpoint(model, weights, config, RECEIPT)
    restored = load_checkpoint(weights, config, trunk_artifact=trunk_artifact, synthetic_trunk=True)
    actual = restored.forward_features(sample)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1]["one2one"]["boxes"], actual[1]["one2one"]["boxes"])
    assert restored.detect.cv2[0][1][0] is restored.detect.cv2[0][1][1]
    assert restored.quantization == "fp32_fp16_fp32_v1"

    state = torch.load(weights, weights_only=True)
    assert set(state) == set(model.detect.state_dict())
    assert not any(key.startswith("model.") for key in state)
    aliases = [key for key in state if key.startswith("cv2.0.1.") and key.endswith("conv.weight")]
    assert len(aliases) == 2
    state[aliases[1]] = state[aliases[1]] + 1
    torch.save(state, weights)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_checkpoint(weights, config, trunk_artifact=trunk_artifact, synthetic_trunk=True)

    optimizer = torch.optim.SGD({id(p): p for p in model.parameters() if p.requires_grad}.values(), lr=0.01)
    resume = tmp_path / "resume.pt"
    save_resume(model, optimizer, resume, epoch=4)
    with torch.no_grad():
        next(model.detect.parameters()).add_(1)
    assert load_resume(model, optimizer, resume) == 4
    broken = dict(RECEIPT, trunk_bytes=7)
    with pytest.raises(ValueError, match="allowlisted"):
        save_checkpoint(model, tmp_path / "bad.pt", tmp_path / "bad.json", broken)
    data = json.loads(config.read_text())
    data["quantization"] = "unquantized"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="quantization"):
        load_checkpoint(weights, config, trunk_artifact=trunk_artifact, synthetic_trunk=True)
