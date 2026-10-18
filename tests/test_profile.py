import pytest
import torch
from torch import nn

from head_sharing.model import HeadStudyModel
from head_sharing.profile import (main, measure_conv_macs, measure_cuda_peak, measure_latency,
                                  storage_inventory, _fixed_probe_batch, _load_training_cache, _training_batch_size,
                                  _fused_inference_copy)


def test_cli_help_is_available(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("argparse help did not exit successfully")
    help_text = capsys.readouterr().out
    assert "--device" in help_text
    assert "--measurement" in help_text
    assert "--inference-policy" in help_text
    assert "--weights" in help_text


def test_checkpoint_inventory_separates_head_and_full_detector_files(tmp_path):
    model = HeadStudyModel("native", seed=19, imgsz=512)
    weights = tmp_path / "selected.pt"
    weights.write_bytes(b"selected serialized weights")
    head = storage_inventory(model.detect, weights)
    full = storage_inventory(model.detector)
    assert head["serialized_file_bytes"] == weights.stat().st_size
    assert full["serialized_file_bytes"] is None
    assert head["unique_parameters"] == storage_inventory(model)["unique_parameters"]
    assert full["unique_parameters"] > head["unique_parameters"]


def test_reused_conv_counts_each_execution_and_unique_storage(tmp_path):
    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            conv = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False)
            self.left = conv
            self.right = conv

        def forward(self, x):
            return self.right(self.left(x))

    model = Reused()
    x = torch.ones(1, 2, 4, 4)
    result = measure_conv_macs(model, lambda: model(x))
    one = 1 * 4 * 4 * 2 * 2 * 3 * 3
    assert result["unique_conv_modules"] == 1
    assert result["conv_invocations"] == 2
    assert result["conv_macs"] == 2 * one
    path = tmp_path / "weights.bin"
    path.write_bytes(b"12345")
    footprint = storage_inventory(model, path)
    assert footprint["unique_parameters"] == 36
    assert footprint["unique_parameter_storage_bytes"] == 36 * 4
    assert footprint["serialized_file_bytes"] == 5


def test_native_both_branches_retained_and_alias_invocations_counted():
    native = HeadStudyModel("native", seed=19, imgsz=512).eval()
    shared = HeadStudyModel("shared_k2", seed=19, imgsz=512).eval()
    independent = HeadStudyModel("independent_k2", seed=19, imgsz=512).eval()
    features = [torch.randn(1, c, n, n) for c, n in ((64, 8), (128, 4), (256, 2))]
    shared_counts = {"box": 0, "class": 0}
    handles = []
    for branch in (shared.detect.cv2, shared.detect.one2one_cv2):
        for site in branch:
            handles.append(site[1][0].register_forward_hook(lambda *args: shared_counts.__setitem__("box", shared_counts["box"] + 1)))
    for branch in (shared.detect.cv3, shared.detect.one2one_cv3):
        for site in branch:
            handles.append(site[1][0].register_forward_hook(lambda *args: shared_counts.__setitem__("class", shared_counts["class"] + 1)))
    try:
        native_out = native.forward_features(features)
        native_macs = measure_conv_macs(native, lambda: native.forward_features(features))
        macs = measure_conv_macs(shared, lambda: shared.forward_features(features))
        independent_macs = measure_conv_macs(independent, lambda: independent.forward_features(features))
    finally:
        for handle in handles:
            handle.remove()
    assert set(native_out[1]) == {"one2many", "one2one"}
    assert shared_counts == {"box": 12, "class": 12}
    assert macs["conv_invocations"] > 0
    assert macs["conv_macs"] == independent_macs["conv_macs"] > native_macs["conv_macs"]
    assert storage_inventory(shared)["unique_parameters"] == storage_inventory(native)["unique_parameters"] == 309656
    assert storage_inventory(independent)["unique_parameters"] > storage_inventory(shared)["unique_parameters"]


def test_cuda_peak_synchronizes_baselines_resets_and_reports_both_peaks():
    class FakeCuda:
        def __init__(self):
            self.events = []

        def synchronize(self, device): self.events.append("sync")
        def memory_allocated(self, device): self.events.append("allocated"); return 10
        def memory_reserved(self, device): self.events.append("reserved"); return 20
        def reset_peak_memory_stats(self, device): self.events.append("reset")
        def max_memory_allocated(self, device): self.events.append("peak_allocated"); return 30
        def max_memory_reserved(self, device): self.events.append("peak_reserved"); return 40
        def is_available(self): return True

    class FakeTorch:
        cuda = FakeCuda()

    device = torch.device("cuda")
    measured = measure_cuda_peak(FakeTorch, device, lambda: FakeTorch.cuda.events.append("operation"))
    assert measured == {"baseline_allocated_bytes": 10, "baseline_reserved_bytes": 20,
                       "peak_allocated_bytes": 30, "peak_reserved_bytes": 40}
    assert FakeTorch.cuda.events == ["sync", "allocated", "reserved", "reset", "operation", "sync",
                                     "peak_allocated", "peak_reserved"]
    try:
        measure_cuda_peak(FakeTorch, torch.device("cpu"), lambda: None)
    except RuntimeError as exc:
        assert "requires the explicitly selected CUDA" in str(exc)
    else:
        raise AssertionError("CPU peak measurement was not refused")


def test_training_cache_uses_frozen_producer_not_execution_gpu(tmp_path, monkeypatch):
    from head_sharing import cache
    execution_uuid = "GPU-assignment-0"
    recipe = {"producer": "GPU-common-cache-1", "cache_evidence": {"few": {"metadata_file_sha256": "a" * 64}}}
    observed = {}
    monkeypatch.setattr(cache, "_sha", lambda path, progress_path=None: "a" * 64)
    def load(path, *, expected_producer, progress_path):
        observed.update(path=path, producer=expected_producer, progress=progress_path)
        return "reader"
    monkeypatch.setattr(cache, "load_cache", load)
    assert _load_training_cache(tmp_path, recipe, "progress") == "reader"
    assert observed == {"path": tmp_path, "producer": recipe["producer"], "progress": "progress"}
    assert observed["producer"] != execution_uuid


def test_training_probe_accepts_only_b1_or_frozen_batch():
    assert _training_batch_size(1, 32) == 1
    assert _training_batch_size(32, 32) == 32
    try:
        _training_batch_size(8, 32)
    except ValueError as exc:
        assert "frozen recipe batch_size" in str(exc)
    else:
        raise AssertionError("unsupported intermediate batch was accepted")


def test_probe_repeats_one_complete_seeded_batch_and_skips_partial_tail():
    import pytest
    partial = ([torch.zeros(1, 2)], {'image': 'partial'})
    complete = ([torch.zeros(2, 2)], {'image': 'complete'})
    epochs = []
    def source(epoch):
        epochs.append(epoch)
        yield partial
        yield complete
    batches = _fixed_probe_batch(source, 2)
    assert batches.steps == 1 and epochs == [0]
    assert next(iter(batches(0))) is complete
    assert next(iter(batches(1))) is complete
    assert epochs == [0]
    with pytest.raises(ValueError, match='no complete batch'):
        _fixed_probe_batch(lambda epoch: iter([partial]), 2)


def test_persisted_synthetic_cpu_profile_cli_binds_loaded_checkpoint(tmp_path):
    import hashlib, json
    from head_sharing.checkpoint import save_checkpoint
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION, freeze
    from head_sharing.runtime import GPU_UUIDS

    model = HeadStudyModel("native", seed=22, imgsz=64)
    with torch.no_grad():
        next(model.detect.parameters()).fill_(0.0125)
    artifact = {"schema_version": 1, "architecture": "yolo26n", "upstream": "8.4.177", "excluded_head_index": 23,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.detector.state_dict().items()
                               if k.startswith("model.") and int(k.split(".")[1]) < 23}}
    trunk = tmp_path / "trunk.pt"
    torch.save(artifact, trunk)
    receipt = {"trunk_artifact_sha256": "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016",
               "trunk_tensors": 468, "trunk_bytes": 9109168, "head_index": 23, "input_channels": [64,128,256],
               "source_variant": "U", "source_seed": 11, "source_epoch_zero_based": 79, "source_training_epochs": 80,
               "pre_exposure": "synthetic generated topology fixture"}
    weights, config = tmp_path / "selected.pt", tmp_path / "selected.json"
    save_checkpoint(model, weights, config, receipt)
    saved_head = torch.load(weights, map_location="cpu", weights_only=True)
    changed_name = next(name for name, _ in model.detect.named_parameters())
    assert torch.all(saved_head[changed_name] == 0.0125)
    recipe = {"status":"FROZEN", "seed_list":[11,22,33,44,55], "arms":["native","independent_k2","shared_k2"],
              "imgsz":64,"batch_size":2,"epochs":2,"cadence":1,"block_size":2,"probe_size":2,
              "optimizer":{"name":"SGD","lr":.01,"momentum":.9,"weight_decay":.0005},
              "gains":{"box":7.5,"cls":.5,"dfl":1.5},"quantization":"fp32_fp16_fp32_v1",
              "producer":GPU_UUIDS[1],"environment":{"device":"CPU synthetic"},"trunk_sha256":receipt["trunk_artifact_sha256"],
              "split_sha256":"a"*64,"cache_evidence":{"synthetic":True},"pilot_evidence":{"synthetic":True},
              "synthetic":True,"gpu_for_seed":{str(seed):GPU_UUIDS[0] for seed in (11,22,33,44,55)},
              "selection":SELECTION,"numeric_policy":NUMERIC_POLICY}
    recipe_path=tmp_path/"recipe.json"; protocol=tmp_path/"protocol.json"
    recipe_path.write_text(json.dumps(recipe)); freeze(recipe_path,protocol,synthetic=True)
    output=tmp_path/"profile.json"
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,config,trunk)}
    assert main(["--device","cpu","--synthetic","--measurement","inference","--weights",str(weights),
                 "--config",str(config),"--trunk-artifact",str(trunk),"--protocol",str(protocol),
                 "--imgsz","64","--batch-size","1","--warmup","0","--repetitions","1","--output",str(output)]) == 0
    result=json.loads(output.read_text())
    assert result["seed"] == 22 and result["arm"] == "native"
    assert result["branch_policy"] == "retained_dual"
    assert "post-fuse" in result["portable_weight_files"]["scope"]
    assert "after branch-policy preparation" in result["executable_storage_scope"]
    assert result["model_identity"]["weights_sha256"] == hashes[weights]
    assert result["model_identity"]["recipe_sha256"]
    assert result["storage"]["head"]["serialized_file_bytes"] == weights.stat().st_size
    assert result["storage"]["full_detector"]["serialized_file_bytes"] is None
    portable=result["portable_weight_files"]
    assert portable["head_file_bytes"] == weights.stat().st_size
    assert portable["common_trunk_file_bytes"] == trunk.stat().st_size
    assert portable["head_plus_trunk_file_bytes"] == weights.stat().st_size + trunk.stat().st_size
    assert result["executed_conv_macs"]["conv_macs"] > result["head_executed_conv_macs"]["conv_macs"] > 0
    assert result["full_detector"]["timing_label"] == result["head"]["timing_label"] == "CPU non-study smoke only"
    assert result["inference_peak"] is None and result["checkpoint_files_unchanged"] is True
    assert result["checkpoint_file_sha256"] == {"head_weights":hashes[weights],"head_config":hashes[config],"trunk_artifact":hashes[trunk]}
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes} == hashes

    fused_output = tmp_path / "profile-fused.json"
    assert main(["--device","cpu","--synthetic","--measurement","inference","--inference-policy","fused_one2one",
                 "--weights",str(weights),"--config",str(config),"--trunk-artifact",str(trunk),"--protocol",str(protocol),
                 "--imgsz","64","--batch-size","1","--warmup","0","--repetitions","1","--output",str(fused_output)]) == 0
    fused_result = json.loads(fused_output.read_text())
    assert fused_result["branch_policy"] == "fused_one2one"
    assert fused_result["storage"]["head"]["serialized_file_bytes"] is None
    assert fused_result["storage"]["head"]["unique_parameters"] < result["storage"]["head"]["unique_parameters"]
    assert fused_result["inference_peak"] is None
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes} == hashes


def test_persisted_refinement_profile_binds_format2_spec_and_rejects_mismatch(tmp_path):
    import hashlib, json
    from head_sharing.checkpoint import save_checkpoint
    from head_sharing.head_spec import head_spec_digest
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION, freeze
    from head_sharing.runtime import GPU_UUIDS

    spec = {"version": 1, "variant": "learned_alpha", "class_inner": 96}
    model = HeadStudyModel("loop_ab2", seed=33, imgsz=64, head_spec=spec).eval()
    features = [torch.randn(1, channels, side, side) for channels, side in ((64,8),(128,4),(256,2))]
    with torch.inference_mode():
        retained = model.forward_features(features)
        fused_model = _fused_inference_copy(model, torch)
        deployed = fused_model.forward_features(features)
    assert torch.allclose(retained[0], deployed[0], rtol=1e-6, atol=1e-6)
    for key in ("boxes", "scores"):
        assert torch.allclose(retained[1]["one2one"][key], deployed[1]["one2one"][key], rtol=1e-6, atol=1e-6)
    artifact = {"schema_version": 1, "architecture": "yolo26n", "upstream": "8.4.177", "excluded_head_index": 23,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.detector.state_dict().items()
                               if k.startswith("model.") and int(k.split(".")[1]) < 23}}
    trunk = tmp_path / "trunk.pt"; torch.save(artifact, trunk)
    receipt = {"trunk_artifact_sha256": "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016",
               "trunk_tensors": 468, "trunk_bytes": 9109168, "head_index": 23, "input_channels": [64,128,256],
               "source_variant": "U", "source_seed": 11, "source_epoch_zero_based": 79,
               "source_training_epochs": 80, "pre_exposure": "synthetic generated topology fixture"}
    weights, config = tmp_path / "selected.pt", tmp_path / "selected.json"
    save_checkpoint(model, weights, config, receipt)
    reference = {"protocol": {"path": "source/protocol.json", "sha256": "a"*64, "recipe_sha256": "b"*64},
                 "freeze": {"path": "source/freeze.json", "sha256": "c"*64}}
    recipe = {"status":"FROZEN", "seed_list":[11,33,55], "arms":["loop_ab2"], "imgsz":64,
              "batch_size":2,"epochs":2,"cadence":1,"block_size":2,"probe_size":2,
              "optimizer":{"name":"SGD","lr":.01,"momentum":.9,"weight_decay":.0005},
              "gains":{"box":7.5,"cls":.5,"dfl":1.5},"quantization":"fp32_fp16_fp32_v1",
              "producer":GPU_UUIDS[1],"environment":{"device":"CPU synthetic","study_id":"ab2_refinement_v1",
              "head_spec":spec,"reference_baseline":reference},"trunk_sha256":receipt["trunk_artifact_sha256"],
              "split_sha256":"d"*64,"cache_evidence":{"synthetic":True},"pilot_evidence":{"synthetic":True},
              "synthetic":True,"gpu_for_seed":{str(seed):GPU_UUIDS[0] for seed in (11,33,55)},
              "selection":SELECTION,"numeric_policy":NUMERIC_POLICY}
    recipe_path=tmp_path/"recipe.json"; protocol=tmp_path/"protocol.json"
    recipe_path.write_text(json.dumps(recipe)); freeze(recipe_path,protocol,synthetic=True)
    args = ["--device","cpu","--synthetic","--weights",str(weights),"--config",str(config),
            "--trunk-artifact",str(trunk),"--protocol",str(protocol),"--imgsz","64","--batch-size","1",
            "--warmup","0","--repetitions","1"]
    output=tmp_path/"profile.json"
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,config,trunk)}
    assert main([*args,"--output",str(output)]) == 0
    result=json.loads(output.read_text())
    assert result["model_identity"]["head_spec"] == spec
    assert result["model_identity"]["head_spec_sha256"] == head_spec_digest(spec)
    assert result["model_identity"]["variant"] == "learned_alpha"
    assert result["model_identity"]["recipe_sha256"]
    assert result["checkpoint_files_unchanged"] is True
    fused_output=tmp_path/"fused.json"
    assert main([*args,"--inference-policy","fused_one2one","--output",str(fused_output)]) == 0
    fused=json.loads(fused_output.read_text())
    assert fused["model_identity"] == result["model_identity"]
    assert fused["branch_policy"] == "fused_one2one"
    before=hashes.copy()
    mismatch={"version":1,"variant":"step_norm","class_inner":96}
    wrong=HeadStudyModel("loop_ab2",seed=33,imgsz=64,head_spec=mismatch)
    save_checkpoint(wrong,weights,config,receipt)
    mismatched_hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,config,trunk)}
    rejected=tmp_path/"rejected.json"
    with pytest.raises(ValueError,match="head_spec differs"):
        main([*args,"--output",str(rejected)])
    assert not rejected.exists()
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,config,trunk)} == mismatched_hashes
    assert before[trunk] == mismatched_hashes[trunk]
    wrong_seed = HeadStudyModel("loop_ab2", seed=22, imgsz=64, head_spec=spec)
    save_checkpoint(wrong_seed,weights,config,receipt)
    seed_hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,config,trunk)}
    with pytest.raises(ValueError,match="arm, seed, or head_spec"):
        main([*args,"--output",str(tmp_path/"wrong-seed.json")])
    assert not (tmp_path/"wrong-seed.json").exists()
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,config,trunk)} == seed_hashes


def test_looped_profile_fuse_preserves_selected_outputs_and_counts_real_calls():
    expected_macs = {"native":384606208,"loop_ab2":631357440,"loop_a3":636919808}
    expected_calls = {"native":48,"loop_ab2":44,"loop_a3":40}
    for arm in ("native", "loop_ab2", "loop_a3"):
        retained = HeadStudyModel(arm, seed=11, imgsz=512).eval()
        features = [torch.randn(1, channels, side, side) for channels, side in ((64,64),(128,32),(256,16))]
        with torch.no_grad():
            retained_output = retained.forward_features(features)
            macs = measure_conv_macs(retained.detect, lambda: retained.forward_features(features))
        original_state = {k:v.clone() for k,v in retained.detect.state_dict().items()}
        retained.train()
        fused = _fused_inference_copy(retained, torch).eval()
        assert retained.training  # policy preparation must not change the caller's mode
        assert all(torch.equal(v, original_state[k]) for k,v in retained.detect.state_dict().items())
        retained.eval()
        with torch.no_grad():
            fused_output = fused.forward_features(features)
        assert macs["conv_macs"] == expected_macs[arm]
        assert macs["conv_invocations"] == expected_calls[arm]
        assert torch.allclose(retained_output[0], fused_output[0], rtol=1e-6, atol=1e-6)
        for key in ("boxes", "scores"):
            assert torch.allclose(retained_output[1]["one2one"][key], fused_output[1]["one2one"][key],
                                  rtol=1e-6, atol=1e-6)
        assert fused.detect.cv2 is fused.detect.cv3 is None
        assert retained.detect.cv2 is not None and retained.detect.cv3 is not None
        fused_macs = measure_conv_macs(fused.detect, lambda: fused.forward_features(features))
        assert fused_macs["conv_macs"] * 2 == expected_macs[arm]
        assert fused_macs["conv_invocations"] * 2 == expected_calls[arm]


def test_training_policy_rejects_fused_before_protocol_or_filesystem_work(tmp_path, monkeypatch):
    from head_sharing.profile import main
    import head_sharing.protocol as protocol
    called = []
    monkeypatch.setattr(protocol, "verify_frozen", lambda *args, **kwargs: called.append(True))
    with pytest.raises(SystemExit):
        main(["--measurement","training","--inference-policy","fused_one2one","--weights","missing",
              "--config","missing","--trunk-artifact","missing","--protocol","missing","--cache","missing",
              "--output",str(tmp_path/"out.json")])
    assert called == [] and not (tmp_path/"out.json").exists()


def _cuda_guard_fixture(tmp_path, monkeypatch, progress_path=None):
    from head_sharing.runtime import GPU_UUIDS, AllocationLedger, initialize

    root = tmp_path / "study"
    root.mkdir()
    (root / "STUDY_OWNER.txt").write_text("test fixture\n")
    ledger_path = initialize(root, "allocation-ledger")
    ledger = AllocationLedger(ledger_path)
    token = ledger.reserve(GPU_UUIDS[0], 120, phase="final")
    job_root = root / "job"
    job_root.mkdir()
    progress = progress_path or job_root / "progress.jsonl"
    ledger.update_job(token, status="running", cwd=str(job_root), progress_file=str(progress))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUIDS[0])
    for key in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR", "TMPDIR", "CUDA_CACHE_PATH",
                "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        path = job_root / key.lower()
        path.mkdir()
        if key == "YOLO_CONFIG_DIR":
            (path / "Ultralytics").mkdir()
        monkeypatch.setenv(key, str(path))
    import head_sharing.supervisor as supervisor
    monkeypatch.setattr(supervisor, "child_entry", lambda *args, **kwargs: None)
    args = ["--device", "cuda", "--measurement", "inference", "--weights", str(root / "weights.pt"),
            "--config", str(root / "config.json"), "--trunk-artifact", str(root / "trunk.pt"),
            "--protocol", str(root / "protocol.json"), "--uuid", GPU_UUIDS[0], "--owned-root", str(root),
            "--ledger", str(ledger_path), "--reservation", token, "--output", str(job_root / "output.json")]
    return args, progress


def test_cuda_profile_guard_accepts_missing_owned_progress_before_native_imports(tmp_path, monkeypatch):
    import head_sharing.protocol as protocol
    from head_sharing.profile import main

    args, progress = _cuda_guard_fixture(tmp_path, monkeypatch)

    class ReachedProtocolVerification(Exception):
        pass

    def stop_at_protocol(*args, **kwargs):
        raise ReachedProtocolVerification

    monkeypatch.setattr(protocol, "verify_frozen", stop_at_protocol)
    try:
        main(args)
    except ReachedProtocolVerification:
        pass
    else:
        raise AssertionError("profile did not reach protocol verification after confinement")
    assert not progress.exists()


@pytest.mark.parametrize("invalid", ["symlink", "directory", "outside"])
def test_cuda_profile_guard_rejects_nonowned_progress_before_protocol(tmp_path, monkeypatch, invalid):
    import head_sharing.protocol as protocol
    from head_sharing.profile import main

    args, progress = _cuda_guard_fixture(tmp_path, monkeypatch)
    from head_sharing.runtime import AllocationLedger
    token = args[args.index("--reservation") + 1]
    ledger = AllocationLedger(args[args.index("--ledger") + 1])
    if invalid == "outside":
        progress = tmp_path / "outside-progress.jsonl"
        ledger.update_job(token, progress_file=str(progress))
    elif invalid == "directory":
        progress.mkdir()
    else:
        progress.symlink_to(tmp_path / "outside-target")
    protocol_calls = []
    monkeypatch.setattr(protocol, "verify_frozen", lambda *a, **k: protocol_calls.append(True))
    try:
        main(args)
    except RuntimeError as exc:
        assert "progress path" in str(exc)
    else:
        raise AssertionError(f"{invalid} supervisor progress path was accepted")
    assert protocol_calls == []


def test_common_q_head_and_full_cpu_smoke_is_labeled_non_study():
    model = HeadStudyModel("native", seed=27, imgsz=512).eval()
    images = torch.zeros(1, 3, 512, 512)
    expected = model.forward_images(images)
    head = measure_latency(model, images, region="head", warmup=0, repetitions=1)
    full = measure_latency(model, images, region="full", warmup=0, repetitions=1)
    assert torch.equal(expected[0], model.forward_features(model.extract_features(images))[0])
    assert head["timing_label"] == full["timing_label"] == "CPU non-study smoke only"
    assert "FP16 roundtrip" in full["quantization_boundary"]
    assert "FP32 Q feature tensors supplied" in head["quantization_boundary"]
    assert full["batch"] == head["batch"] == 1
    assert "conv_macs" not in full
