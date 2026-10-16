#!/usr/bin/env python3
"""Run an explicitly synthetic CPU native-loss/save-resume/dev/probe smoke journey."""
from __future__ import annotations
import argparse, json, os, random, tempfile, time
from pathlib import Path
import numpy as np


def run(root: Path):
    import hashlib
    import torch
    import cv2
    from head_sharing.model import HeadStudyModel
    from head_sharing.train import experiment, metric_annotations
    from head_sharing.checkpoint import load_checkpoint
    from head_sharing.cache import build_cache, _sha
    from head_sharing.data import compact_manifest
    from head_sharing.geometry import COCO80_TO_91
    torch.set_num_threads(1); cv2.setNumThreads(1)
    root.mkdir(parents=True, exist_ok=False)
    images = root/'images'; images.mkdir()
    categories = [{'id': c, 'name': str(c)} for c in COCO80_TO_91]
    mapping = [{'category_id': c, 'class_id': i, 'name': str(c)} for i, c in enumerate(COCO80_TO_91)]
    annotation = {'images': [], 'categories': categories, 'annotations': []}; identities = []
    rng = np.random.default_rng(23)
    for image_id in range(1, 5):
        name = f'{image_id}.png'
        pixels = rng.integers(0, 256, (48+image_id, 60+image_id, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images/name), pixels)
        annotation['images'].append({'id': image_id, 'file_name': name, 'width': pixels.shape[1], 'height': pixels.shape[0]})
        identities.append({'image_id': image_id, 'file_name': name, 'sha256': _sha(images/name)})
        for crowd in (0, 1):
            annotation['annotations'].append({'id': image_id*2+crowd, 'image_id': image_id, 'category_id': 1,
                'bbox': [10, 8, 20, 16], 'area': 320, 'iscrowd': crowd, 'segmentation': [[10,8,30,8,30,24,10,24]]})
    annotations_path = root/'annotations.json'; annotations_path.write_text(json.dumps(annotation))
    source = {'ids': {'few': [1, 2], 'dev': [3, 4]}, 'images': identities, 'categories': mapping,
              'manifest_sha256': hashlib.sha256(b'synthetic source split').hexdigest(),
              'source_annotation_sha256': _sha(annotations_path)}
    manifest = compact_manifest(source, images, probe_ids=[1, 2])
    manifest_path = root/'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    trunk_model = HeadStudyModel('native', 997, 64)
    artifact = {'schema_version': 1, 'architecture': 'yolo26n', 'upstream': '8.4.177', 'excluded_head_index': 23,
                'state_dict': {k: v.clone() for k, v in trunk_model.detector.state_dict().items()
                               if k.startswith('model.') and int(k.split('.')[1]) < 23}}
    torch.save(artifact, root/'synthetic-trunk.pt')
    # The existing cache producer permits generated in-memory trunks ONLY with
    # synthetic=True. This receipt is a topology fixture, never source evidence.
    receipt = {'trunk_artifact_sha256': 'f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016',
               'trunk_tensors': 468, 'trunk_bytes': 9109168, 'head_index': 23, 'input_channels': [64,128,256],
               'source_variant': 'U', 'source_seed': 11, 'source_epoch_zero_based': 79, 'source_training_epochs': 80,
               'pre_exposure': 'SYNTHETIC topology fixture; generated random trunk, NOT study source'}
    receipt_path = root/'receipt.json'; receipt_path.write_text(json.dumps(receipt))
    paths = {}; evidence = {}
    for cohort in ('few', 'dev', 'probe'):
        paths[cohort] = root/('cache-'+cohort)
        metadata = build_cache(image_root=images, annotations_path=annotations_path, manifest_path=manifest_path,
                               receipt_path=receipt_path, trunk_artifact=artifact, output=paths[cohort], model=trunk_model,
                               producer='cpu', cohort=cohort, batch_size=2, synthetic=True)
        evidence[cohort] = {'metadata_file_sha256': _sha(paths[cohort]/'metadata.json')}
    gt = metric_annotations(annotations_path, source['source_annotation_sha256'], [1,2,3,4])
    assert sum(x['iscrowd'] for x in gt['annotations']) == 4
    assert all('area' in x and 'segmentation' not in x for x in gt['annotations'])
    recipe = {'seed_list': [11,22,33,44,55], 'arms': ['native','independent_k2','shared_k2'], 'imgsz': 64,
              'epochs': 2, 'cadence': 1, 'batch_size': 2, 'producer': 'cpu', 'synthetic': True,
              'quantization': 'fp32_fp16_fp32_v1', 'split_sha256': _sha(manifest_path),
              'trunk_sha256': receipt['trunk_artifact_sha256'], 'cache_evidence': evidence}
    from head_sharing.protocol import freeze, verify_frozen, SELECTION, NUMERIC_POLICY
    from head_sharing.runtime import GPU_UUIDS
    recipe.update(status='FROZEN', block_size=2, probe_size=2, optimizer={'name':'SGD','lr':.01,'momentum':.9,'weight_decay':.0005},
                  gains={'box':7.5,'cls':.5,'dfl':1.5}, environment={'device':'CPU synthetic fixture'},
                  pilot_evidence={'synthetic':True, 'purpose':'generated fixture'},
                  gpu_for_seed={str(seed): GPU_UUIDS[0] for seed in recipe['seed_list']},
                  selection=SELECTION, numeric_policy=NUMERIC_POLICY)
    recipe_path = root/'recipe.json'; recipe_path.write_text(json.dumps(recipe))
    frozen_path = root/'protocol.json'; freeze(recipe_path, frozen_path, synthetic=True)
    protocol = verify_frozen(frozen_path, synthetic=True)
    recipe = protocol['recipe']; protocol_sha = protocol['recipe_sha256']
    def equal(a, b):
        if isinstance(a, torch.Tensor): assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal(x, y)
        else: assert a == b
    outputs = []
    for arm in recipe['arms']:
        common = dict(purpose='main', arm=arm, seed=11, recipe=recipe, protocol_sha256=protocol_sha,
                      cache_paths=paths, manifest_path=manifest_path, annotations_path=annotations_path,
                      trunk_artifact=torch.load(root/'synthetic-trunk.pt', weights_only=True), receipt=receipt,
                      device='cpu', synthetic=True)
        uninterrupted = root/(arm+'-full'); interrupted = root/(arm+'-resume')
        if arm == 'native':
            from head_sharing.train import main as train_main
            train_main(['run', '--device', 'cpu', '--synthetic', '--arm', arm, '--seed', '11',
                        '--cache', str(paths['few']), '--dev-cache', str(paths['dev']), '--probe-cache', str(paths['probe']),
                        '--manifest', str(manifest_path), '--annotations', str(annotations_path), '--protocol', str(frozen_path),
                        '--receipt', str(receipt_path), '--trunk-artifact', str(root/'synthetic-trunk.pt'), '--output', str(uninterrupted)])
            full = json.loads((uninterrupted/'training.json').read_text())
        else: full = experiment(**common, output=uninterrupted)
        first = experiment(**common, output=interrupted, stop_epoch=1)
        boundary = torch.load(interrupted/'resume.pt', weights_only=True)
        assert boundary['epoch'] == 1 and 'criterion' not in boundary['training_state']
        resumed_root = root/(arm+'-continued')
        boundary_hash = _sha(interrupted/'resume.pt')
        resumed = experiment(**common, output=resumed_root, resume_from=interrupted/'resume.pt')
        assert _sha(interrupted/'resume.pt') == boundary_hash
        if arm == 'native':
            for field, replacement in (('protocol_sha256', 'wrong-protocol'), ('recipe', dict(recipe, epochs=3))):
                malformed = dict(common); malformed[field] = replacement
                bad_output = root/('bad-'+field)
                try: experiment(**malformed, output=bad_output, resume_from=interrupted/'resume.pt')
                except ValueError: pass
                else: raise AssertionError('malformed immutable resume binding accepted')
                assert not bad_output.exists() and _sha(interrupted/'resume.pt') == boundary_hash
            malformed = dict(common, recipe=dict(recipe, split_sha256=manifest['manifest_sha256']))
            try: experiment(**malformed, output=root/'bad-semantic-file-sha')
            except ValueError: pass
            else: raise AssertionError('semantic source digest accepted as compact FILE SHA')
        a = torch.load(uninterrupted/'resume.pt', weights_only=True)
        b = torch.load(resumed_root/'resume.pt', weights_only=True)
        for key in ('model', 'optimizer', 'rng', 'epoch'): equal(a[key], b[key])
        for x, y in zip(full['history'], resumed['history']):
            for key in ('loss', 'o2m', 'lr', 'dev', 'probe', 'gap'): equal(x[key], y[key])
        assert full['frozen_trunk_digest'] == resumed['frozen_trunk_digest']
        assert full['head_updated'] and resumed['head_updated']
        assert full['last_head_parameter_sha256'] == resumed['last_head_parameter_sha256']
        aps = [row['dev']['AP50:95'] for row in resumed['history']]
        assert resumed['selected_epoch'] == aps.index(max(aps))+1
        fresh = HeadStudyModel(arm, 11, 64)
        changed = any(not torch.equal(v, fresh.detect.state_dict()[k]) for k,v in b['model'].items())
        assert changed and resumed['epochs_completed'] == 2
        for config_name in (resumed['selected_config'], resumed['last_config']):
            config = resumed_root/config_name
            loaded = load_checkpoint(config.with_suffix('.pt'), config, trunk_artifact=artifact, synthetic_trunk=True)
            with torch.no_grad(): assert torch.isfinite(loaded.forward_images(torch.zeros(2,3,64,64))[0]).all()
        for cached, record in ((False, full), (True, resumed)):
            outputs.append({'arm': arm, 'cached': True, 'reloaded': cached, 'native_loss': record['history'][0]['loss'],
                            'changed': changed, 'resume_epoch': 1, 'completed': 2})
    from head_sharing.train import main as train_main
    train_main(['pilot', '--device', 'cpu', '--synthetic', '--arm', 'native', '--seed', '997', '--imgsz', '64',
                '--epochs', '1', '--batch-size', '2', '--producer', 'cpu', '--cache', str(paths['few']),
                '--manifest', str(manifest_path), '--trunk-artifact', str(root/'synthetic-trunk.pt'),
                '--receipt', str(receipt_path), '--output', str(root/'technical-pilot')])
    pilot = json.loads((root/'technical-pilot'/'training.json').read_text())
    assert pilot['purpose'] == 'pilot' and pilot['protocol_sha256'] is None and pilot['selected_config'] is None and pilot['head_updated']
    assert all('dev' not in row and 'probe' not in row for row in pilot['history'])
    report = {'synthetic': True, 'purpose': 'generated persisted-cache native lifecycle; NOT study evidence',
              'arms': outputs, 'dev_coco': resumed['history'][-1]['dev'], 'probe_coco': resumed['history'][-1]['probe'],
              'target_epochs': 2, 'completed_epochs': 2, 'resume_exact': True,
              'fixture_trunk_file_sha256': _sha(root/'synthetic-trunk.pt')}
    (root/'synthetic-journey.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",required=True); a=p.parse_args(argv)
    root=Path(a.root).absolute()
    if not os.environ.get("HEAD_SHARING_TEST_ROOT") or not root.resolve(strict=False).is_relative_to(Path(os.environ["HEAD_SHARING_TEST_ROOT"]).resolve(strict=True)):
        raise RuntimeError("synthetic output must be under the isolated test runner root")
    import socket
    def denied(*args, **kwargs): raise RuntimeError('network denied in synthetic fixture worker')
    socket.create_connection = denied; socket.getaddrinfo = denied; socket.socket.connect = denied
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    result=run(root)
    print(json.dumps({"synthetic":True,"runs":len(result["arms"]),"completed_epochs":result["completed_epochs"],"dev_AP":result["dev_coco"]["AP50:95"]},sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
