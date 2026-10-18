import copy
import json

import pytest


def recipe(spec=None):
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION
    return {
        "status": "FROZEN", "seed_list": [11, 22, 33, 44, 55], "arms": ["native"],
        "imgsz": 64, "batch_size": 2, "epochs": 2, "cadence": 1, "block_size": 2,
        "probe_size": 2, "optimizer": {"name": "SGD", "lr": .01, "momentum": .9, "weight_decay": .0005},
        "gains": {"box": 7.5, "cls": .5, "dfl": 1.5}, "quantization": "fp32_fp16_fp32_v1",
        "producer": "cpu", "environment": {"study_id": "shared_spatial_v1", "head_spec": spec},
        "trunk_sha256": "a" * 64, "split_sha256": "b" * 64,
        "cache_evidence": {"few": "synthetic", "dev": "synthetic", "probe": "synthetic"},
        "pilot_evidence": {"synthetic": True}, "synthetic": True,
        "gpu_for_seed": {str(seed): "GPU-08055392-87d6-8287-947d-0bf94c711445" for seed in [11,22,33,44,55]},
        "selection": copy.deepcopy(SELECTION), "numeric_policy": copy.deepcopy(NUMERIC_POLICY),
    }


def test_shared_spatial_recipe_requires_exact_new_matrix_and_canonical_v2_or_none():
    from head_sharing.protocol import validate_recipe
    from head_sharing.head_spec import canonicalize_head_spec
    for spec in (None, canonicalize_head_spec({"version": 2, "variant": "spatial_gn"})):
        value = recipe(spec)
        assert validate_recipe(value, synthetic=True) is value
    invalid = [
        lambda r: r["seed_list"].pop(),
        lambda r: r.update(arms=["independent_k2"]),
        lambda r: r.update(epochs=41),
        lambda r: r["environment"].update(head_spec={"version": 1, "variant": "baseline", "class_inner": 96}),
        lambda r: r["environment"].update(head_spec={"version": 2, "variant": "unregistered"}),
        lambda r: r["environment"].pop("head_spec"),
    ]
    for mutate in invalid:
        value = recipe()
        mutate(value)
        with pytest.raises(ValueError):
            validate_recipe(value, synthetic=True)
    production = recipe()
    production.update(imgsz=512, batch_size=32, block_size=32, epochs=40, cadence=5, probe_size=1000, synthetic=False)
    production["environment"]["head_spec"] = None
    production["producer"] = "GPU-08055392-87d6-8287-947d-0bf94c711445"
    production["gpu_for_seed"] = {str(seed): "GPU-08055392-87d6-8287-947d-0bf94c711445" for seed in [11,22,33,44,55]}
    production["cache_evidence"] = {key: {"metadata_file_sha256": "c" * 64} for key in ("few", "dev", "probe")}
    # Production recipes are structurally checked without touching live artifacts.
    assert validate_recipe(production, synthetic=False) is production
    wrong_producer = copy.deepcopy(production)
    wrong_producer["producer"] = "GPU-13916c7e-aa3c-fa5e-48ef-7b8cca10ea99"
    with pytest.raises(ValueError, match="producer"):
        validate_recipe(wrong_producer, synthetic=False)
    wrong_assignment = copy.deepcopy(production)
    wrong_assignment["gpu_for_seed"]["11"] = "GPU-13916c7e-aa3c-fa5e-48ef-7b8cca10ea99"
    with pytest.raises(ValueError, match="G1"):
        validate_recipe(wrong_assignment, synthetic=False)
    overlong = copy.deepcopy(production)
    overlong["epochs"] = 80
    with pytest.raises(ValueError, match="dose"):
        validate_recipe(overlong, synthetic=False)
    synthetic_cpu = recipe()
    assert validate_recipe(synthetic_cpu, synthetic=True) is synthetic_cpu


@pytest.mark.parametrize("spec", [None, {"version": 2, "variant": "spatial_sepbn"}])
def test_synthetic_fit_persisted_resume_and_recovery(tmp_path, spec):
    import hashlib
    import torch
    from head_sharing import cache
    from head_sharing.model import HeadStudyModel
    from head_sharing.protocol import freeze
    from head_sharing.runtime import GPU_UUIDS
    from head_sharing.train import experiment
    from test_cache import RECEIPT, fixture

    torch.set_num_threads(1)
    root = tmp_path / "study"
    images, annotations, manifest_path, receipt_path, _, _ = fixture(root / "inputs")
    source = json.loads(annotations.read_text())
    for row in source["annotations"]:
        row["area"] = row["bbox"][2] * row["bbox"][3]
    annotations.write_text(json.dumps(source))
    manifest = json.loads(manifest_path.read_text())
    manifest["source_annotation_sha256"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    trunk_model = HeadStudyModel("native", seed=123, imgsz=64).eval()
    artifact = {"state_dict": {k: v.detach().cpu() for k,v in trunk_model.detector.state_dict().items()
                                if k.startswith("model.") and int(k.split(".")[1]) < 23},
                "schema_version":1,"architecture":"yolo26n","upstream":"8.4.177","excluded_head_index":23}
    artifact_path = root / "trunk.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    paths, evidence = {}, {}
    for cohort in ("few", "dev", "probe"):
        path = root / ("cache-" + cohort)
        cache.build_cache(image_root=images, annotations_path=annotations, manifest_path=manifest_path,
                          receipt_path=receipt_path, trunk_artifact=artifact, output=path,
                          model=trunk_model, producer="cpu", cohort=cohort, batch_size=2, synthetic=True)
        paths[cohort] = path
        evidence[cohort] = {"metadata_file_sha256": cache._sha(path / "metadata.json")}
    value = recipe(spec)
    value.update(producer="cpu", cache_evidence=evidence, split_sha256=cache._sha(manifest_path),
                 trunk_sha256=RECEIPT["trunk_artifact_sha256"])
    protocol_path, recipe_path = root / "protocol.json", root / "recipe.json"
    recipe_path.write_text(json.dumps(value))
    frozen = freeze(recipe_path, protocol_path, synthetic=True)
    common = dict(purpose="main", arm="native", seed=11, recipe=value, protocol_sha256=frozen["recipe_sha256"],
                  cache_paths=paths, manifest_path=manifest_path, annotations_path=annotations,
                  trunk_artifact=artifact, receipt=RECEIPT, device="cpu", synthetic=True)
    complete = experiment(**common, output=root/"complete")
    interrupted_path = root / "interrupted"
    experiment(**common, output=interrupted_path, stop_epoch=1)
    checkpoint_before = cache._sha(interrupted_path/"resume.pt")
    rng_before = torch.get_rng_state().clone()
    wrong_spec = {"version": 2, "variant": "spatial_sepbn" if spec is None else "spatial_untied"}
    wrong_recipe = copy.deepcopy(value)
    wrong_recipe["environment"]["head_spec"] = wrong_spec
    with pytest.raises(ValueError, match="head_spec identity mismatch"):
        experiment(**{**common, "recipe": wrong_recipe}, output=interrupted_path, resume=True)
    assert cache._sha(interrupted_path/"resume.pt") == checkpoint_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    resumed = experiment(**common, output=interrupted_path, resume=True)
    assert complete["recipe"] == resumed["recipe"] == value
    assert complete["head_spec"] == resumed["head_spec"] == spec
    assert complete["head_spec_sha256"] == resumed["head_spec_sha256"]
    assert complete["epochs_completed"] == resumed["epochs_completed"] == 2
    assert all("dev" in row and "probe" in row for row in resumed["history"])
    assert json.loads((interrupted_path/"training.json").read_text())["head_spec_sha256"] == complete["head_spec_sha256"]
    state_a = torch.load(root/"complete"/"resume.pt", map_location="cpu", weights_only=True)
    state_b = torch.load(interrupted_path/"resume.pt", map_location="cpu", weights_only=True)
    assert state_a["training_state"]["recipe"] == state_b["training_state"]["recipe"] == value
    assert state_a["model"].keys() == state_b["model"].keys()
    assert all(torch.equal(state_a["model"][key], state_b["model"][key]) for key in state_a["model"])
    resume_hash = cache._sha(root/"complete"/"resume.pt")
    presentation = json.loads((root/"complete"/"training.json").read_text())
    presentation["epochs_completed"] = 1
    (root/"complete"/"training.json").write_text(json.dumps(presentation))
    repaired = experiment(**common, output=root/"complete", resume=True)
    assert repaired["recipe"] == value and repaired["head_spec"] == spec
    assert cache._sha(root/"complete"/"resume.pt") == resume_hash


@pytest.mark.parametrize("spec", [None, {"version": 2, "variant": "spatial_sepbn"}])
def test_profile_cli_binds_shared_spatial_recipe_spec(tmp_path, spec):
    import hashlib
    import torch
    from head_sharing.checkpoint import save_checkpoint
    from head_sharing.profile import main
    from head_sharing.protocol import freeze
    from head_sharing.runtime import GPU_UUIDS

    from head_sharing.model import HeadStudyModel
    model = HeadStudyModel("native", seed=22, imgsz=64, head_spec=spec).eval()
    artifact = {"schema_version": 1, "architecture": "yolo26n", "upstream": "8.4.177", "excluded_head_index": 23,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.detector.state_dict().items()
                               if k.startswith("model.") and int(k.split(".")[1]) < 23}}
    trunk = tmp_path / "trunk.pt"
    torch.save(artifact, trunk)
    receipt = {"trunk_artifact_sha256": "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016",
               "trunk_tensors": 468, "trunk_bytes": 9109168, "head_index": 23, "input_channels": [64,128,256],
               "source_variant": "U", "source_seed": 11, "source_epoch_zero_based": 79,
               "source_training_epochs": 80, "pre_exposure": "synthetic generated topology fixture"}
    weights, config = tmp_path / "selected.pt", tmp_path / "selected.json"
    save_checkpoint(model, weights, config, receipt)
    value = recipe(spec)
    value["environment"]["device"] = "CPU synthetic"
    value["producer"] = GPU_UUIDS[1]
    value["gpu_for_seed"] = {str(seed): GPU_UUIDS[1] for seed in value["seed_list"]}
    protocol_path, recipe_path = tmp_path / "protocol.json", tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(value))
    freeze(recipe_path, protocol_path, synthetic=True)
    output = tmp_path / "profile.json"
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (weights, config, trunk)}
    assert main(["--device", "cpu", "--synthetic", "--weights", str(weights), "--config", str(config),
                 "--trunk-artifact", str(trunk), "--protocol", str(protocol_path), "--imgsz", "64",
                 "--batch-size", "1", "--warmup", "0", "--repetitions", "1", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    from head_sharing.head_spec import head_spec_digest
    assert result["model_identity"]["variant"] == ("native" if spec is None else spec["variant"])
    assert result["model_identity"]["head_spec"] == spec
    assert result["model_identity"]["head_spec_sha256"] == head_spec_digest(spec)
    assert result["model_identity"]["recipe_sha256"]
    assert result["checkpoint_files_unchanged"] is True
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes} == source_hashes


def test_new_study_finite_gradient_observation_is_pre_step_and_failure_context(tmp_path):
    import torch
    from torch import nn
    from types import SimpleNamespace
    from head_sharing.train import train

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.detect = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
        def extract_features(self, x):
            return x
        def forward_features(self, x):
            return self.detect(x)
    class Criterion:
        one2one = SimpleNamespace(hyp=SimpleNamespace(epochs=1))
        o2m = "synthetic"
        def __init__(self, bad=False, huge_gradient=False): self.bad = bad; self.huge_gradient = huge_gradient
        def __call__(self, prediction, batch):
            if self.bad:
                return torch.stack((prediction.sum(), prediction.sum() * torch.tensor(float("nan")))), {}
            if self.huge_gradient:
                return prediction.sum() * 1e30, {}
            return torch.stack((prediction.sum(), prediction.sum() * 0 + 1)), {}
        def update(self): pass
    def batches(_epoch):
        yield torch.ones(2,1,2,2), {}
    batches.steps = 1
    model = Model()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    train(model, batches, epochs=1, seed=11, criterion=Criterion(), finite_diagnostics=True)
    assert any(not torch.equal(old, new) for old, new in zip(initial, model.parameters()))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(FloatingPointError, match='non_finite_loss'): 
        train(model, batches, epochs=1, seed=11, criterion=Criterion(True), finite_diagnostics=True,
              diagnostic_context={"study_id":"shared_spatial_v1", "variant":"native", "seed":11,
                                  "failure_path":str(tmp_path/"nonfinite.json")})
    diagnostic = json.loads((tmp_path/"nonfinite.json").read_text())
    assert diagnostic == {"epoch":1,"failure_kind":"non_finite_loss","loss":"non-finite","phase":"loss","seed":11,
                          "step":1,"study_id":"shared_spatial_v1","variant":"native"}
    with pytest.raises(FloatingPointError):
        train(model, batches, epochs=1, seed=11, criterion=Criterion(True))
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
    with pytest.raises(FloatingPointError, match="non_finite_gradient_norm"):
        train(model, batches, epochs=1, seed=11, criterion=Criterion(huge_gradient=True), finite_diagnostics=True,
              diagnostic_context={"study_id":"shared_spatial_v1", "variant":"native", "seed":11,
                                  "failure_path":str(tmp_path/"nonfinite-gradient.json")})
    gradient_diagnostic = json.loads((tmp_path/"nonfinite-gradient.json").read_text())
    assert gradient_diagnostic["phase"] == "backward" and gradient_diagnostic["failure_kind"] == "non_finite_gradient_norm"
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
