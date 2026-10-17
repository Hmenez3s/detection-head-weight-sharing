import gzip
import hashlib
import json
import os
import subprocess
import sys

import pytest


def _synthetic_study(root, arms=None):
    import cv2
    import numpy as np
    import torch
    from head_sharing.checkpoint import save_checkpoint, save_resume
    from head_sharing.evaluate import ARMS, LOOPED_MATRIX, POLICY, SEEDS, _sha, freeze_selected
    from head_sharing.model import HeadStudyModel
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION, freeze as freeze_protocol
    from head_sharing.runtime import GPU_UUIDS

    root.mkdir(parents=True)
    (root / 'STUDY_OWNER.txt').write_text('synthetic fixture only\n')
    receipt = {'trunk_artifact_sha256': 'f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016',
               'trunk_tensors': 468, 'trunk_bytes': 9109168, 'head_index': 23,
               'input_channels': [64, 128, 256], 'source_variant': 'U', 'source_seed': 11,
               'source_epoch_zero_based': 79, 'source_training_epochs': 80,
               'pre_exposure': 'SYNTHETIC generated topology fixture; NOT study evidence'}
    trunk_model = HeadStudyModel('native', 11, 64)
    trunk = {'schema_version': 1, 'architecture': 'yolo26n', 'upstream': '8.4.177', 'excluded_head_index': 23,
             'state_dict': {k: v.clone() for k, v in trunk_model.detector.state_dict().items()
                            if k.startswith('model.') and int(k.split('.')[1]) < 23}}
    trunk_path = root / 'synthetic-trunk.pt'; torch.save(trunk, trunk_path)
    selected_arms = tuple(arms or ARMS)
    looped = selected_arms == tuple(LOOPED_MATRIX)
    source_fixture = _synthetic_study(root / 'source', ARMS) if looped else None
    reference = ({'protocol': {'path':'source/protocol.json','sha256':_sha(source_fixture['protocol_path']),
                               'recipe_sha256':source_fixture['protocol']['recipe_sha256']},
                  'freeze': {'path':'source/freeze.json','sha256':_sha(source_fixture['frozen_path'])}}
                 if looped else None)
    recipe = {'status': 'FROZEN', 'seed_list': list(SEEDS), 'arms': list(selected_arms), 'imgsz': 64,
              'batch_size': 2, 'epochs': 10, 'cadence': 5, 'block_size': 2, 'probe_size': 2,
              'optimizer': {'name': 'SGD', 'lr': .01, 'momentum': .9, 'weight_decay': .0005},
              'gains': {'box': 7.5, 'cls': .5, 'dfl': 1.5}, 'quantization': 'fp32_fp16_fp32_v1',
              'producer': 'cpu', 'environment': ({'device': 'CPU synthetic fixture'} if not looped else {
                  'study_id':'looped_v1', 'looped_architecture':__import__('head_sharing.protocol',fromlist=['LOOPED_ARCHITECTURE']).LOOPED_ARCHITECTURE,
                  'reference_native':reference}),
              'trunk_sha256': receipt['trunk_artifact_sha256'], 'split_sha256': 'a' * 64,
              'cache_evidence': {'synthetic': True}, 'pilot_evidence': {'synthetic': True},
              'synthetic': True, 'gpu_for_seed': {str(seed): GPU_UUIDS[0] for seed in SEEDS},
              'selection': SELECTION, 'numeric_policy': NUMERIC_POLICY}
    protocol_path = root / 'protocol.json'
    freeze_protocol_path = root / 'recipe.json'; freeze_protocol_path.write_text(json.dumps(recipe))
    protocol = freeze_protocol(freeze_protocol_path, protocol_path, synthetic=True)
    runs = []
    run_arms = tuple(__import__('head_sharing',fromlist=['LOOPED_ARMS']).LOOPED_ARMS) if looped else selected_arms
    for arm in run_arms:
        for seed in SEEDS:
            model = HeadStudyModel(arm, seed, 64)
            model.detector.load_state_dict({k: v for k, v in trunk_model.detector.state_dict().items()
                                            if k in model.detector.state_dict()}, strict=False)
            run = root / f'{seed}-{arm}'; run.mkdir(); runs.append(run)
            cfg, weights = run / 'selected-e5.json', run / 'selected-e5.pt'
            save_checkpoint(model, weights, cfg, receipt)
            histories = [{'epoch': epoch, 'loss': 1.0} for epoch in range(1, 11)]
            histories[4]['dev'] = {'AP50:95': .25}; histories[9]['dev'] = {'AP50:95': .25}
            hashes = {cfg.name: _sha(cfg), weights.name: _sha(weights)}
            state = {'step': 10, 'total_steps': 10, 'history': histories,
                     'selected_epoch': 5, 'selected_dev_ap': .25, 'selected_hashes': hashes,
                     'purpose': 'main', 'protocol_sha256': protocol['recipe_sha256'], 'arm': arm,
                     'seed': seed, 'target_epochs': 10, 'synthetic': True, 'recipe': recipe,
                     'trunk_digest': 'synthetic-trunk-state'}
            optimizer = torch.optim.SGD(model.detect.parameters(), lr=.01)
            save_resume(model, optimizer, run / 'resume.pt', epoch=10, training_state=state)
            record = {'schema_version': 1, 'purpose': 'main', 'synthetic': True, 'arm': arm, 'seed': seed,
                      'protocol_sha256': protocol['recipe_sha256'], 'target_epochs': 10, 'epochs_completed': 10,
                      'selected_epoch': 5, 'selected_dev_ap': .25, 'selection_metric': 'dev.AP50:95',
                      'frozen_trunk_digest': 'synthetic-trunk-state', 'selected_config': cfg.name,
                      'history': histories}
            (run / 'training.json').write_text(json.dumps(record))
    frozen_path = root / 'freeze.json'
    frozen = freeze_selected(protocol_path, runs, frozen_path, root)
    images = root / 'images'; images.mkdir()
    ann = {'images': [], 'categories': [{'id': 1, 'name': 'person'}], 'annotations': []}
    for image_id in (1, 2, 3):
        pixels = np.full((64, 64, 3), image_id * 20, dtype=np.uint8)
        assert cv2.imwrite(str(images / f'{image_id}.png'), pixels)
        ann['images'].append({'id': image_id, 'file_name': f'{image_id}.png', 'width': 64, 'height': 64})
        ann['annotations'].append({'id': image_id, 'image_id': image_id, 'category_id': 1,
                                   'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0})
    annotations = root / 'annotations.json'; annotations.write_text(json.dumps(ann))
    ids = root / 'ids.json'; ids.write_text(json.dumps({'image_ids': [1, 2, 3]}))
    receipt_path = root / 'receipt.json'; receipt_path.write_text(json.dumps(receipt))
    return locals()


def test_evaluate_help_is_lazy_and_lists_evidence_lifecycle():
    proc = subprocess.run(
        [sys.executable, '-c', "import sys; import head_sharing.evaluate; assert 'torch' not in sys.modules; head_sharing.evaluate.main(['--help'])"],
        env={**os.environ, 'PYTHONPATH': 'src'}, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert 'freeze-selected' in proc.stdout and 'predict' in proc.stdout and 'score' in proc.stdout


def test_full15_synthetic_freeze_member_predictions_and_score_replay(tmp_path, monkeypatch):
    from head_sharing.evaluate import _sha, predict, score_prediction, verify_freeze
    fixture = _synthetic_study(tmp_path / 'study')
    root, frozen_path, frozen = fixture['root'], fixture['frozen_path'], fixture['frozen']
    verified = verify_freeze(frozen_path, root)
    assert len(verified['members']) == 15
    assert all(row['selected_epoch'] == 5 for row in verified['members'])
    saved_freeze = frozen_path.read_bytes()
    frozen['members'].pop()
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='complete'):
        verify_freeze(frozen_path, root)
    frozen_path.write_bytes(saved_freeze)
    progress = root / 'progress.jsonl'
    outputs = []
    monkeypatch.setenv('HEAD_SHARING_CPU_BUDGET', '2')
    for seed, arm in ((11, 'native'), (22, 'shared_k2')):
        output = root / f'{seed}-{arm}.json.gz'
        result = predict(frozen_path, seed=seed, arm=arm, owned_root=root,
                         annotations_path=fixture['annotations'], image_root=fixture['images'],
                         image_ids_path=fixture['ids'], trunk_artifact=fixture['trunk_path'],
                         receipt_path=fixture['receipt_path'], output=output, synthetic=True,
                         progress_path=progress)
        assert result['seed'] == seed and result['arm'] == arm and isinstance(result['count'], int)
        import cv2, torch
        assert torch.get_num_threads() == 2 and cv2.getNumThreads() == 1
        monkeypatch.setattr('head_sharing.evaluate.predict', lambda *args, **kwargs: pytest.fail('score invoked inference'))
        replay = score_prediction(frozen_path, owned_root=root, annotations_path=fixture['annotations'],
                                  predictions_path=output)
        from head_sharing.coco import score
        with gzip.open(output, 'rt') as stream: payload = json.load(stream)
        direct = score(json.loads(fixture['annotations'].read_text()), payload['image_ids'], payload['predictions'])
        assert replay['AP50:95'] == direct['AP50:95']
        assert (replay['seed'], replay['arm']) == (seed, arm)
        assert replay['execution_uuid'] == 'cpu'
        assert replay['prediction_sha256'] == _sha(output)
        assert replay['freeze_sha256'] == _sha(frozen_path)
        outputs.append(output)
    assert len({_sha(path.with_suffix(path.suffix + '.json')) for path in outputs}) == 2
    progress_rows = [json.loads(line) for line in progress.read_text().splitlines()]
    assert any(row['phase'] == 'prediction_batches' and row['done'] == row['total'] == 3
               for row in progress_rows)
    with pytest.raises(FileExistsError):
        predict(frozen_path, seed=11, arm='native', owned_root=root,
                annotations_path=fixture['annotations'], image_root=fixture['images'],
                image_ids_path=fixture['ids'], trunk_artifact=fixture['trunk_path'],
                receipt_path=fixture['receipt_path'], output=outputs[0], synthetic=True)
    ann_original = fixture['annotations'].read_bytes()
    fixture['annotations'].write_bytes(ann_original + b' ')
    with pytest.raises(ValueError, match='annotation'):
        score_prediction(frozen_path, owned_root=root, annotations_path=fixture['annotations'], predictions_path=outputs[0])
    fixture['annotations'].write_bytes(ann_original)
    with gzip.open(outputs[0], 'rt') as stream: payload = json.load(stream)
    payload['predictions'] = payload['predictions'][:-1]
    with gzip.open(outputs[0], 'wt') as stream: json.dump(payload, stream)
    with pytest.raises(ValueError):
        score_prediction(frozen_path, owned_root=root, annotations_path=fixture['annotations'], predictions_path=outputs[0])


def test_looped_matrix_freezes_and_predicts_actual_new_arm(tmp_path):
    from head_sharing.evaluate import predict, verify_freeze
    from head_sharing.protocol import LOOPED_MATRIX
    fixture=_synthetic_study(tmp_path/'study', LOOPED_MATRIX)
    frozen=verify_freeze(fixture['frozen_path'],fixture['root'])
    assert [row['arm'] for row in frozen['members'][:3]]==list(LOOPED_MATRIX)
    assert all(row['source']['kind']=='current' for row in frozen['members'] if row['arm']!='native')
    assert all(row['source']['kind']=='reused_native' for row in frozen['members'] if row['arm']=='native')
    with pytest.raises(ValueError,match='original prediction identity'):
        predict(fixture['frozen_path'],seed=11,arm='native',owned_root=fixture['root'],
                annotations_path=fixture['annotations'],image_root=fixture['images'],image_ids_path=fixture['ids'],
                trunk_artifact=fixture['trunk_path'],receipt_path=fixture['receipt_path'],
                output=fixture['root']/'blocked-native.gz',synthetic=True)
    source=fixture['source_fixture']
    original_prediction=source['root']/'native-original.gz'
    predict(source['frozen_path'],seed=11,arm='native',owned_root=source['root'],
            annotations_path=source['annotations'],image_root=source['images'],image_ids_path=source['ids'],
            trunk_artifact=source['trunk_path'],receipt_path=source['receipt_path'],
            output=original_prediction,synthetic=True)
    from head_sharing.evaluate import score_prediction, member_source_binding
    source_native=next(row for row in frozen['members'] if row['seed']==11 and row['arm']=='native')
    binding=member_source_binding(frozen,source_native,'new-freeze-file-sha')
    assert binding=={'protocol_sha256':source['protocol']['recipe_sha256'],
                     'freeze_sha256':fixture['reference']['freeze']['sha256']}
    replay=score_prediction(fixture['frozen_path'],owned_root=fixture['root'],
                            annotations_path=source['annotations'],predictions_path=original_prediction)
    assert replay['freeze_sha256']==fixture['reference']['freeze']['sha256']
    assert replay['protocol_sha256']==source['protocol']['recipe_sha256']
    import shutil
    for source_run in source['runs']:
        if source_run.name.split('-',1)[1] != 'native': shutil.rmtree(source_run)
    assert len(verify_freeze(fixture['frozen_path'],fixture['root'])['members'])==15
    native_weights=fixture['root']/source_native['weights']
    saved_weights=native_weights.read_bytes(); native_weights.write_bytes(saved_weights+b'tampered')
    with pytest.raises(ValueError,match='digest'):
        predict(fixture['frozen_path'],seed=22,arm='loop_a3',owned_root=fixture['root'],
                annotations_path=fixture['root']/'missing-annotations.json',image_root=fixture['root'],
                image_ids_path=fixture['root']/'missing-ids.json',trunk_artifact=fixture['trunk_path'],
                receipt_path=fixture['receipt_path'],output=fixture['root']/'blocked-weight.gz',synthetic=True)
    native_weights.write_bytes(saved_weights)
    original_freeze=fixture['frozen_path'].read_bytes()
    sentinel=fixture['root']/'not-read'
    for mode in ('missing','swapped','tampered'):
        tampered=json.loads(original_freeze)
        native=next(row for row in tampered['members'] if row['seed']==11 and row['arm']=='native')
        if mode=='missing': native.pop('source')
        elif mode=='swapped': native['source']=next(row['source'] for row in tampered['members'] if row['arm']=='loop_a3')
        else: native['source']['freeze_sha256']='0'*64
        fixture['frozen_path'].write_text(json.dumps(tampered))
        with pytest.raises(ValueError):
            predict(fixture['frozen_path'],seed=22,arm='loop_a3',owned_root=fixture['root'],
                    annotations_path=sentinel,image_root=sentinel,image_ids_path=sentinel,
                    trunk_artifact=sentinel,receipt_path=sentinel,output=fixture['root']/f'blocked-{mode}.gz',synthetic=True)
    fixture['frozen_path'].write_bytes(original_freeze)
    output=fixture['root']/'loop-a3.gz'
    result=predict(fixture['frozen_path'],seed=11,arm='loop_a3',owned_root=fixture['root'],
                   annotations_path=fixture['annotations'],image_root=fixture['images'],image_ids_path=fixture['ids'],
                   trunk_artifact=fixture['trunk_path'],receipt_path=fixture['receipt_path'],output=output,synthetic=True)
    assert result['arm']=='loop_a3' and result['count']>=0
    replay_new=score_prediction(fixture['frozen_path'],owned_root=fixture['root'],
                                annotations_path=fixture['annotations'],predictions_path=output)
    with gzip.open(output,'rt') as stream: new_payload=json.load(stream)
    from head_sharing.coco import score as direct_coco_score
    direct=direct_coco_score(json.loads(fixture['annotations'].read_text()),new_payload['image_ids'],new_payload['predictions'])
    assert replay_new['AP50:95']==direct['AP50:95']
    assert replay_new['freeze_sha256']==__import__('head_sharing.evaluate',fromlist=['_sha'])._sha(fixture['frozen_path'])
    assert replay_new['protocol_sha256']==fixture['protocol']['recipe_sha256']


def test_freeze_rejects_incomplete_and_mutated_selected_weights_before_holdout_open(tmp_path):
    from head_sharing.evaluate import freeze_selected, predict
    fixture = _synthetic_study(tmp_path / 'study')
    root, run = fixture['root'], fixture['runs'][0]
    sentinel = root / 'sentinel'; sentinel.write_text('not opened')
    run_weights = run / 'selected-e5.pt'
    run_weights.write_bytes(run_weights.read_bytes() + b'change')
    with pytest.raises(ValueError, match='digest'):
        freeze_selected(fixture['protocol_path'], fixture['runs'], root / 'bad-freeze.json', root)
    with pytest.raises(ValueError, match='changed'):
        predict(fixture['frozen_path'], seed=11, arm='native', owned_root=root,
                annotations_path=sentinel, image_root=sentinel, image_ids_path=sentinel,
                trunk_artifact=sentinel, receipt_path=sentinel, output=root / 'blocked.gz', synthetic=True)


def test_score_only_does_not_import_inference_stack(tmp_path):
    from head_sharing.evaluate import predict
    fixture = _synthetic_study(tmp_path/'lazy-score')
    prediction = fixture['root']/'native-predictions.gz'
    predict(fixture['frozen_path'], seed=11, arm='native', owned_root=fixture['root'],
            annotations_path=fixture['annotations'], image_root=fixture['images'], image_ids_path=fixture['ids'],
            trunk_artifact=fixture['trunk_path'], receipt_path=fixture['receipt_path'],
            output=prediction, synthetic=True)
    code = ("import sys; from head_sharing.evaluate import score_prediction, member_source_binding; "
            "assert 'torch' not in sys.modules; "
            f"result=score_prediction({str(fixture['frozen_path'])!r},owned_root={str(fixture['root'])!r},"
            f"annotations_path={str(fixture['annotations'])!r},predictions_path={str(prediction)!r}); "
            "assert 'AP50:95' in result; assert 'torch' not in sys.modules; "
            "assert 'head_sharing.model' not in sys.modules; assert 'ultralytics' not in sys.modules")
    proc = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': 'src'},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
