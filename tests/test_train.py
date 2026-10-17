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
    with __import__("pytest").raises(ValueError): cache_batches(Reader(),seed=11,ids=[10],batch_size=2)


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
