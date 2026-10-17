import json
import os
import subprocess
import sys


def test_cache_targets_use_image_row_indices_for_multiple_objects(monkeypatch):
    import torch
    from head_sharing import train as module
    from head_sharing.train import cache_batches
    rows=[{"targets_xywhn":[[0,.1,.2,.3,.4],[1,.2,.3,.4,.5]],"flipped":{"targets_xywhn":[[0,.1,.2,.3,.4],[1,.2,.3,.4,.5]]}},
          {"targets_xywhn":[[2,.3,.4,.5,.6],[3,.4,.5,.6,.7]],"flipped":{"targets_xywhn":[[2,.3,.4,.5,.6],[3,.4,.5,.6,.7]]}}]
    class Reader:
        metadata={"fixed_layout":{"ids":[10,20],"blocks":[[10,20]]},"block_order":{"few":[10,20]},"cohort_ids":{"few":[10,20]},"cohort":"few"}
    monkeypatch.setattr("head_sharing.cache.read_blocks",lambda *a,**k: iter([([torch.zeros(2,1,1,1)]*3,rows)]))
    batches=cache_batches(Reader(),seed=11,ids=[10,20],batch_size=2)
    _,targets=next(batches(0))
    assert targets["batch_idx"].tolist()==[0.,0.,1.,1.]
    assert batches.steps == 1
    with __import__("pytest").raises(ValueError): cache_batches(Reader(),seed=11,ids=[10],batch_size=2)


def test_cache_smaller_batches_keep_all_images_and_one_block_flip(monkeypatch):
    import torch
    from head_sharing.train import cache_batches
    rows = [{"targets_xywhn": [[i,.1,.2,.3,.4]],
             "flipped": {"targets_xywhn": [[i,.9,.2,.3,.4]]}} for i in range(2)]
    class Reader:
        metadata={"fixed_layout":{"ids":[10,20],"blocks":[[10,20]]},"block_order":{"few":[10,20]},"cohort_ids":{"few":[10,20]},"cohort":"few"}
    monkeypatch.setattr('head_sharing.data.epoch_batches', lambda layout,seed,epoch: iter([([10,20],True)]))
    calls = []
    def read(reader, cohort, group, *, flipped, batch_size):
        calls.append((group,flipped,batch_size))
        for i,row in enumerate(rows):
            yield [torch.full((1,1,1,1),float(i))]*3,[row]
    monkeypatch.setattr('head_sharing.cache.read_blocks',read)
    batches = cache_batches(Reader(),seed=11,ids=[10,20],batch_size=1)
    got = list(batches(0))
    assert batches.steps == len(got) == 2
    assert calls == [([10,20],True,1)]
    assert [features[0].item() for features,_ in got] == [0.,1.]
    assert [target['cls'].item() for _,target in got] == [0.,1.]
    assert all(target['batch_idx'].tolist() == [0.] for _,target in got)
    assert all(target['bboxes'][0,0].item() == __import__('pytest').approx(.9) for _,target in got)
    with __import__('pytest').raises(ValueError,match='positive'):
        cache_batches(Reader(),seed=11,ids=[10,20],batch_size=0)


def test_train_cli_help_does_not_import_native_runtime():
    proc=subprocess.run([sys.executable,"-c","import sys; import head_sharing.train; assert 'torch' not in sys.modules; head_sharing.train.main(['--help'])"],env={**os.environ,"PYTHONPATH":"src"},stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=20)
    assert proc.returncode==0,proc.stderr


def test_synthetic_journey_runs_under_isolated_cpu_wrapper():
    from head_sharing.supervisor import _terminate_owned, _pid_identity
    proc=subprocess.Popen([sys.executable,"scripts/synthetic_journey.py","--root",os.path.join(os.environ["HEAD_SHARING_TEST_ROOT"],"journey")],
                          stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    identity = _pid_identity(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=240)
    finally:
        _terminate_owned(proc, identity)
    assert proc.returncode==0, stderr
    result=json.loads(stdout.splitlines()[-1])
    assert result["synthetic"] is True and result["runs"]==6 and result["completed_epochs"]==2
    report=json.load(open(os.path.join(os.environ["HEAD_SHARING_TEST_ROOT"],"journey","synthetic-journey.json")))
    assert len(report["arms"])==6 and report['resume_exact'] is True
    assert all(row["native_loss"]>0 and row["changed"] and row["resume_epoch"]==1 for row in report["arms"])
    assert 0.0<=report["dev_coco"]["AP50:95"]<=1.0
    for arm in ('native', 'independent_k2', 'shared_k2'):
        record=json.load(open(os.path.join(os.environ['HEAD_SHARING_TEST_ROOT'], 'journey', arm+'-continued', 'training.json')))
        assert record['schema_version']==1 and record['purpose']=='main'
        assert record['epochs_completed']==record['target_epochs']==2
        assert record['selected_config'] and record['last_config'] == 'last.json'
        assert record['numeric_policy']['deterministic_algorithms'] is False
        assert record['numeric_policy']['cudnn_deterministic'] is False
        assert all('dev' in row and 'probe' in row and 'gap' in row for row in record['history'])
        assert not any(os.environ['HEAD_SHARING_TEST_ROOT'] in str(value) for value in record.values())


def test_cli_rejects_unsupervised_gpu_and_non_synthetic_cpu_before_native_imports():
    base=['pilot', '--arm', 'native', '--seed', '997', '--cache', 'missing', '--output', 'missing-output',
          '--manifest', 'missing-manifest', '--receipt', 'missing-receipt', '--trunk-artifact', 'missing-trunk',
          '--imgsz', '640', '--epochs', '2', '--batch-size', '32', '--producer', 'producer']
    code = """
import sys
from head_sharing.train import main
for device in ('cpu', 'GPU-13916c7e-aa3c-fa5e-48ef-7b8cca10ea99'):
    try: main(%r+['--device', device])
    except (RuntimeError, SystemExit): pass
    else: raise AssertionError('unguarded native worker accepted')
    assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
""" % base
    proc=subprocess.run([sys.executable, '-c', code], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr


def test_completed_checkpoint_repairs_failed_final_presentation(tmp_path, monkeypatch):
    import hashlib
    import pytest
    import torch
    import head_sharing.cache as cache
    from head_sharing import train as train_module
    from head_sharing.model import HeadStudyModel
    from head_sharing.checkpoint import load_checkpoint, load_resume, load_trunk_artifact, _rng_state
    from head_sharing.protocol import SELECTION, NUMERIC_POLICY
    from head_sharing.runtime import GPU_UUIDS
    from test_cache import RECEIPT, fixture

    torch.set_num_threads(1)
    images, annotations, manifest_path, receipt_path, trunk_model, artifact = fixture(tmp_path / "fixture")
    source_annotations = json.loads(annotations.read_text())
    for row in source_annotations["annotations"]:
        row["area"] = row["bbox"][2] * row["bbox"][3]
    annotations.write_text(json.dumps(source_annotations))
    manifest = json.loads(manifest_path.read_text())
    manifest["source_annotation_sha256"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    cache_paths = {}
    evidence = {}
    for cohort in ("few", "dev", "probe"):
        path = tmp_path / f"cache-{cohort}"
        cache.build_cache(image_root=images, annotations_path=annotations, manifest_path=manifest_path,
                          receipt_path=receipt_path, trunk_artifact=artifact, output=path,
                          model=trunk_model, producer="cpu", cohort=cohort, batch_size=2,
                          synthetic=True)
        cache_paths[cohort] = path
        evidence[cohort] = {"metadata_file_sha256": cache._sha(path / "metadata.json")}
    recipe = {"seed_list": [11], "arms": ["native"], "imgsz": 32, "epochs": 2, "cadence": 1,
              "batch_size": 2, "producer": "cpu", "synthetic": True,
              "quantization": "fp32_fp16_fp32_v1", "split_sha256": cache._sha(manifest_path),
              "trunk_sha256": RECEIPT["trunk_artifact_sha256"], "cache_evidence": evidence,
              "gpu_for_seed": {"11": GPU_UUIDS[0]}, "selection": SELECTION,
              "numeric_policy": NUMERIC_POLICY}
    common = dict(purpose="main", arm="native", seed=11, recipe=recipe, protocol_sha256="synthetic-protocol",
                  cache_paths=cache_paths, manifest_path=manifest_path, annotations_path=annotations,
                  trunk_artifact=artifact, receipt=RECEIPT, device="cpu", synthetic=True)
    from pathlib import Path
    atomic_json = cache._atomic_json
    original_train = train_module.train

    def fail_final_record(path, value):
        if Path(path).name == "training.json" and value["epochs_completed"] == 2:
            raise OSError("injected final presentation failure")
        return atomic_json(path, value)

    monkeypatch.setattr(cache, "_atomic_json", fail_final_record)
    for recovery_mode in ("inplace", "fresh"):
        output = tmp_path / f"interrupted-{recovery_mode}"
        with pytest.raises(OSError, match="injected final presentation failure"):
            train_module.experiment(**common, output=output)
        checkpoint = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
        stale = json.loads((output / "training.json").read_text())
        assert checkpoint["epoch"] == 2 and len(checkpoint["training_state"]["history"]) == 2
        assert stale["epochs_completed"] == 1 and len(stale["history"]) == 1
        checkpoint_sha = cache._sha(output / "resume.pt")
        selected_hashes = checkpoint["training_state"]["selected_hashes"]
        final_assets = ["resume.pt", "last.pt", "last.json", *selected_hashes]
        asset_hashes = {name: cache._sha(output / name) for name in final_assets}
        monkeypatch.setattr(cache, "_atomic_json", atomic_json)
        destination = output if recovery_mode == "inplace" else tmp_path / "fresh-recovery"
        no_training = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("completed resume trained again"))
        monkeypatch.setattr(train_module, "train", no_training)
        if recovery_mode == "inplace":
            recovered = train_module.experiment(**common, output=destination, resume=True)
        else:
            recovered = train_module.experiment(**common, output=destination, resume_from=output / "resume.pt")
        monkeypatch.setattr(train_module, "train", original_train)
        record = json.loads((destination / "training.json").read_text())
        assert recovered == record
        actual_rng = _rng_state()
        assert torch.equal(actual_rng["torch_cpu"], checkpoint["rng"]["torch_cpu"])
        assert actual_rng["python"] == checkpoint["rng"]["python"]
        assert actual_rng["numpy"][0] == checkpoint["rng"]["numpy"][0]
        assert torch.equal(actual_rng["numpy"][1], checkpoint["rng"]["numpy"][1])
        assert actual_rng["numpy"][2:] == checkpoint["rng"]["numpy"][2:]
        assert record["epochs_completed"] == record["target_epochs"] == 2
        assert record["history"] == checkpoint["training_state"]["history"]
        assert record["selected_epoch"] == checkpoint["training_state"]["selected_epoch"]
        assert record["selected_dev_ap"] == checkpoint["training_state"]["selected_dev_ap"]
        assert record["frozen_trunk_digest"] == stale["frozen_trunk_digest"]
        restored_model = HeadStudyModel("native", 11, 32)
        load_trunk_artifact(restored_model, artifact, receipt=RECEIPT, synthetic=True)
        restored_optimizer = torch.optim.SGD(train_module.optimizer_groups(restored_model, .0005), lr=.01, momentum=.9)
        load_resume(restored_model, restored_optimizer, output / "resume.pt")
        assert record["last_head_parameter_sha256"] == train_module._head_parameter_digest(restored_model)
        assert record["last_head_parameter_sha256"] != stale["last_head_parameter_sha256"]
        assert record["selected_config"] == stale["selected_config"]
        assert {name: cache._sha(destination / name) for name in final_assets} == asset_hashes
        assert {name: cache._sha(output / name) for name in final_assets} == asset_hashes
        last_model = load_checkpoint(destination / "last.pt", destination / "last.json",
                                     trunk_artifact=artifact, synthetic_trunk=True)
        assert train_module._head_parameter_digest(last_model) == record["last_head_parameter_sha256"]
        assert all(torch.equal(last_model.detect.state_dict()[name], restored_model.detect.state_dict()[name])
                   for name in last_model.detect.state_dict())
        assert cache._sha(output / "resume.pt") == checkpoint_sha
        record_sha = cache._sha(destination / "training.json")
        if recovery_mode == "inplace":
            with pytest.raises(ValueError, match="completed runs are immutable"):
                train_module.experiment(**common, output=destination, resume=True)
        else:
            with pytest.raises(FileExistsError):
                train_module.experiment(**common, output=destination)
        assert cache._sha(destination / "training.json") == record_sha
        monkeypatch.setattr(train_module, "train", original_train)
        monkeypatch.setattr(cache, "_atomic_json", fail_final_record)

    missing_record_output = tmp_path / "missing-presentation-baseline"
    with pytest.raises(OSError, match="injected final presentation failure"):
        train_module.experiment(**common, output=missing_record_output)
    (missing_record_output / "training.json").unlink()
    missing_record_checkpoint_sha = cache._sha(missing_record_output / "resume.pt")
    monkeypatch.setattr(cache, "_atomic_json", atomic_json)
    monkeypatch.setattr(train_module, "train", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("completed resume trained again")))
    with pytest.raises(ValueError, match="completed checkpoint lacks presentation baseline"):
        train_module.experiment(**common, output=missing_record_output, resume=True)
    assert not (missing_record_output / "training.json").exists()
    assert cache._sha(missing_record_output / "resume.pt") == missing_record_checkpoint_sha
    monkeypatch.setattr(train_module, "train", original_train)


def test_metric_gt_preserves_original_values_without_borrowing_source_heap(tmp_path, monkeypatch):
    import hashlib
    import head_sharing.cache as cache
    from head_sharing.train import metric_annotations
    source = {'images': [{'id': 1}, {'id': 2}], 'categories': [{'id': 17, 'name': 'example'}],
              'annotations': [{'id': 8, 'image_id': 1, 'category_id': 17, 'iscrowd': 1,
                  'area': 1234.5678901234567, 'bbox': [10.123456789012345, 20.25, 30.5, 40.75],
                  'segmentation': [[1.0, 2.0, 3.0, 4.0]]}]}
    path = tmp_path/'source.json'
    path.write_text(json.dumps(source))
    monkeypatch.setattr(cache, '_read_json', lambda p: source)
    result = metric_annotations(path, hashlib.sha256(path.read_bytes()).hexdigest(), [1])
    expected = {'images': [source['images'][0]], 'categories': source['categories'],
                'annotations': [{k: v for k, v in source['annotations'][0].items() if k != 'segmentation'}]}
    assert result == expected
    assert result['images'][0] is not source['images'][0]
    assert result['annotations'][0]['bbox'][0] is not source['annotations'][0]['bbox'][0]
    assert result['annotations'][0]['iscrowd'] == 1
    assert result['annotations'][0]['area'] == source['annotations'][0]['area']
