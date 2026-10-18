"""Fixed-length native-loss optimization for the frozen-trunk head study."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import ALL_ARMS


def optimizer_groups(model, weight_decay: float):
    import torch
    from torch import nn
    norms = {id(p) for module in model.detect.modules()
             if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm))
             for p in module.parameters(recurse=False)}
    seen, decay, no_decay = set(), [], []
    for name, parameter in model.detect.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") or id(parameter) in norms else decay).append(parameter)
    if not decay or not no_decay:
        raise ValueError("expected convolution weights and norm/bias parameters")
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _seed(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _native_criterion(model, epochs: int):
    from ultralytics.utils.loss import E2ELoss
    # E2ELoss reads detector.args before construction; initialize all native loss gains here.
    model.detector.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=epochs)
    criterion = model.detector.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise RuntimeError("native end-to-end criterion was not constructed")
    return criterion


def train(model, batches, *, epochs: int, seed: int, lr: float = .01,
          momentum: float = .9, weight_decay: float = .0005, start_epoch: int = 0,
          optimizer=None, criterion=None, device: str | torch.device = "cpu", progress=None, stop_epoch: int | None = None,
          batch_progress=None, restored_rng: bool = False):
    """Train on an iterable factory yielding (Q-features or images, batch-targets)."""
    import torch
    if epochs < 1 or not 0 <= start_epoch <= epochs: raise ValueError("invalid fixed epoch range")
    if device == "cpu":
        budget = int(__import__("os").environ.get("HEAD_SHARING_CPU_BUDGET", "3"))
        torch.set_num_threads(max(1, min(3, budget)))
    if start_epoch == 0 and not restored_rng: _seed(seed)
    model.to(device); model.train()
    groups = optimizer_groups(model, weight_decay)
    optimizer = optimizer or torch.optim.SGD(groups, lr=lr, momentum=momentum)
    fresh_criterion = criterion is None
    criterion = criterion or _native_criterion(model, epochs)
    if criterion.one2one.hyp.epochs != epochs: raise ValueError("criterion epoch target differs from fixed target")
    if fresh_criterion:
        for _ in range(start_epoch): criterion.update()
    stop = epochs if stop_epoch is None else stop_epoch
    if not start_epoch <= stop <= epochs: raise ValueError("stop epoch is outside fixed target")
    total_steps = max(1, epochs * max(1, getattr(batches, "steps", 1)))
    step = start_epoch * max(1, getattr(batches, "steps", 1)); losses=[]; state=[]
    for epoch in range(start_epoch, stop):
        model.train(); epoch_losses=[]
        for inputs, batch in batches(epoch):
            if isinstance(inputs, (list, tuple)) and len(inputs) == 3 and inputs[0].ndim == 4 and inputs[0].shape[1] != 3:
                inputs=[x.to(device=device, dtype=torch.float32) for x in inputs]
                # BF16 applies to head operations only; cached Q inputs stay FP32.
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                    predictions=model.forward_features(inputs)
            else:
                images=inputs.to(device=device, dtype=torch.float32)
                # Trunk extraction is FP32/autocast-disabled; autocast wraps only Detect head work.
                features=model.extract_features(images)
                with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.device.type == "cuda"):
                    predictions=model.forward_features(features)
            batch={k:v.to(device) for k,v in batch.items()}
            # Loss modules contain native BCE and gains; keep them under the same
            # BF16 autocast policy as Detect while BCE's own policy promotes FP32.
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                loss, _ = criterion(predictions, batch)
            if not torch.isfinite(loss).all(): raise FloatingPointError("non-finite native loss")
            total_loss=loss.sum()
            frac=step/max(1,total_steps-1); warm=min(1.0,(step+1)/max(1,int(total_steps*.05)))
            rate=lr*warm*.5*(1+math.cos(math.pi*frac))
            for group in optimizer.param_groups: group["lr"]=rate
            optimizer.zero_grad(set_to_none=True); total_loss.backward(); optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu())); step+=1
            if batch_progress: batch_progress({'epoch': epoch+1, 'step': step, 'loss': epoch_losses[-1], 'lr': rate})
        if not epoch_losses: raise ValueError("epoch produced no batches")
        criterion.update()  # native E2ELoss schedule advances once per completed epoch
        losses.append(float(np.mean(epoch_losses)))
        state.append({"epoch":epoch+1,"loss":losses[-1],"o2m":criterion.o2m,"lr":optimizer.param_groups[0]["lr"]})
        if progress: progress(state[-1])
    model.eval()
    return {"model":model,"optimizer":optimizer,"criterion":criterion,"losses":losses,"epochs_completed":stop,"history":state,"step":step,"total_steps":total_steps,"seed":seed}


def cache_batches(reader, *, seed: int, ids: list[int], batch_size: int):
    import torch
    from .cache import read_blocks
    from .data import epoch_batches
    layout=reader.metadata["fixed_layout"]
    if batch_size < 1: raise ValueError("batch_size must be positive")
    if list(ids) != reader.metadata["cohort_ids"].get("few") or layout.get("ids") != reader.metadata["block_order"].get("few"):
        raise ValueError("training IDs must exactly match the bound few cohort and layout")
    def batches(epoch):
        for group, flipped in epoch_batches(layout, seed, epoch):
            # Layout groups are fixed physical cache blocks; only their epoch order changes.
            for features, metadata in read_blocks(reader,reader.metadata["cohort"],group,flipped=flipped,batch_size=batch_size):
                selected=[]
                for image_index, row in enumerate(metadata):
                    target_key="targets_xywhn" if not flipped else "flipped"
                    target=row[target_key] if not flipped else row[target_key]["targets_xywhn"]
                    selected.extend([[image_index,*t] for t in target])
                targets=torch.tensor(selected,dtype=torch.float32).reshape(-1,6)
                yield features,{"batch_idx":targets[:,0],"cls":targets[:,1:2],"bboxes":targets[:,2:]}
    batches.steps=sum((len(group)+batch_size-1)//batch_size for group in layout["blocks"]); return batches


def save_training_state(path, result, *, epoch, **bindings):
    from .checkpoint import save_resume
    # Criterion instances are native objects, not portable state. Replay update()
    # once per completed epoch against the immutable target on fresh construction.
    state={key:result[key] for key in ("step", "total_steps", "seed") if key in result}
    state.update(bindings)
    save_resume(result["model"],result["optimizer"],path,epoch=epoch,training_state=state)


def _trunk_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.detector.model[:-1].state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _head_parameter_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.detect.named_parameters():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metric_annotations(path, expected_sha256, ids):
    """Retain original bbox/crowd/area GT, never transformed cache targets."""
    from .cache import _sha, _read_json
    if _sha(Path(path)) != expected_sha256: raise ValueError('source annotation SHA256 mismatch')
    source = _read_json(Path(path)); wanted = set(ids)
    result = {'images': [x for x in source['images'] if x['id'] in wanted],
              'categories': source['categories'],
              'annotations': [{k: v for k, v in x.items() if k != 'segmentation'}
                              for x in source['annotations'] if x['image_id'] in wanted]}
    if {x['id'] for x in result['images']} != wanted: raise ValueError('metric GT IDs missing')
    # Pack the small retained GT while the full JSON is still alive. Borrowed
    # bbox floats otherwise pin segmentation-sized CPython allocator arenas,
    # evicting the common feature cache despite deleting the source dictionary.
    result = json.loads(json.dumps(result))
    del source
    return result


def evaluate_cache(model, reader, annotation, *, device, progress_path=None):
    import torch
    from .cache import read_blocks, _progress
    from .geometry import Transform, decoded_rows
    from .coco import score
    m = reader.metadata; cohort = m['cohort']; ids = m['block_order'][cohort]
    category_map = [x['category_id'] for x in sorted(m['categories'], key=lambda x: x['class_id'])]
    model.float().eval(); predictions = []; done = 0
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for features, rows in read_blocks(reader, cohort, ids, batch_size=m['batch_size']):
            decoded = model.forward_features([x.to(device) for x in features])[0]
            for output, row in zip(decoded, rows, strict=True):
                geo = Transform(np.empty((m['imgsz'], m['imgsz'], 0), dtype=np.uint8),
                                tuple(row['ratio']), tuple(row['pad']), tuple(row['original_shape']), False)
                predictions.extend(decoded_rows(output, row['image_id'], category_map=category_map, meta=geo))
            done += len(rows); _progress(progress_path, 'evaluate_'+cohort, done, len(ids))
    return score(annotation, m['cohort_ids'][cohort], predictions)


def experiment(*, purpose, arm, seed, recipe, protocol_sha256, cache_paths, manifest_path,
               annotations_path, trunk_artifact, receipt, output, device, synthetic=False,
               resume=False, resume_from=None, progress_path=None, stop_epoch=None, head_spec=None):
    """One fixed-target lifecycle; also the persisted-fixture CPU acceptance path.

    cache_evidence maps cohort to an object containing metadata_file_sha256.
    split_sha256 binds the compact manifest FILE, not its source semantic digest.
    stop_epoch is an API-only interruption oracle; public runs always finish.
    """
    import torch
    from contextlib import ExitStack
    from .cache import load_cache, _read_json, _sha, _atomic_json, _progress
    from .data import fixed_blocks
    from .model import HeadStudyModel
    from .head_spec import canonicalize_head_spec, head_spec_digest
    from .checkpoint import load_trunk_artifact, save_checkpoint, load_checkpoint, load_resume, resume_training_state, _rng_state, _restore_rng
    if purpose not in ('pilot', 'main'): raise ValueError('unknown experiment purpose')
    if purpose == 'pilot' and seed != 997: raise ValueError('technical pilot requires seed 997')
    if purpose == 'main' and (seed not in recipe['seed_list'] or arm not in recipe['arms']):
        raise ValueError('run is outside frozen matrix')
    if bool(recipe['synthetic']) != synthetic: raise ValueError('synthetic recipe mismatch')
    refinement = purpose == 'main' and recipe.get('environment', {}).get('study_id') == 'ab2_refinement_v1'
    if purpose == 'main':
        declared_spec = recipe.get('environment', {}).get('head_spec') if refinement else None
        if head_spec is not None:
            raise ValueError('main head_spec is frozen by the recipe')
        resolved_spec = canonicalize_head_spec(declared_spec)
    else:
        resolved_spec = canonicalize_head_spec(head_spec)
    target = recipe['epochs']; cadence = recipe['cadence']; size = recipe['imgsz']; batch = recipe['batch_size']
    if type(target) is not int or target < 1 or not 1 <= cadence <= target: raise ValueError('invalid target/cadence')
    manifest = _read_json(Path(manifest_path))
    manifest_file_sha = _sha(Path(manifest_path))
    if manifest_file_sha != recipe['split_sha256']: raise ValueError('compact manifest FILE SHA mismatch')
    if recipe['quantization'] != HeadStudyModel.quantization: raise ValueError('recipe Q mismatch')
    if recipe['trunk_sha256'] != receipt['trunk_artifact_sha256']: raise ValueError('recipe trunk mismatch')
    cohorts = {'few': manifest['ids']['few'], 'dev': manifest['ids']['dev'], 'probe': manifest['probe_ids']}
    names = ('few',) if purpose == 'pilot' else ('few', 'dev', 'probe')
    if set(cache_paths) != set(names): raise ValueError('only the declared experiment cohorts are allowed')
    if purpose == 'main' and not synthetic and len(cohorts['probe']) != 1000: raise ValueError('fixed probe must have 1000 IDs')
    output = Path(output)
    if resume and resume_from is not None: raise ValueError('choose in-place or fresh-job resume, not both')
    continuing = resume or resume_from is not None
    resume_path = Path(resume_from) if resume_from is not None else output/'resume.pt'
    if continuing:
        if not resume_path.is_file(): raise ValueError('owned resume checkpoint missing')
        if resume_from is not None and output.exists(): raise FileExistsError('resumed job output must be fresh')
        previous_record = resume_path.parent/'training.json'
        if previous_record.exists() and _read_json(previous_record)['epochs_completed'] == target:
            raise ValueError('completed runs are immutable')
    elif output.exists(): raise FileExistsError('experiment output must be fresh')
    timings = {}; stage_started = time.monotonic()
    with ExitStack() as stack:
        readers = {}
        for name in names:
            expected = {'cohort': name, 'cohort_ids': cohorts, 'imgsz': size, 'batch_size': batch,
                        'manifest_sha256': manifest['manifest_sha256'], 'trunk_sha256': recipe['trunk_sha256'],
                        'source_annotation_sha256': manifest['source_annotation_sha256'], 'categories': manifest['categories'],
                        'fixed_layout': fixed_blocks(cohorts[name], batch), 'count': len(cohorts[name]), 'synthetic': synthetic}
            reader = stack.enter_context(load_cache(cache_paths[name], expected_producer=recipe['producer'],
                                         expected_identity=expected, progress_path=progress_path))
            if [row['image_id'] for row in reader.metadata['images']] != expected['fixed_layout']['ids']:
                raise ValueError('cache rows do not match exact physical cohort order')
            evidence = recipe['cache_evidence'][name]
            if not isinstance(evidence, dict) or _sha(Path(cache_paths[name])/'metadata.json') != evidence.get('metadata_file_sha256'):
                raise ValueError('frozen cache metadata FILE hash mismatch')
            readers[name] = reader
        timings['cache_validation_seconds'] = time.monotonic()-stage_started
        stage_started = time.monotonic(); gt = None
        if purpose == 'main':
            if not annotations_path: raise ValueError('main requires original COCO annotations')
            gt = metric_annotations(annotations_path, manifest['source_annotation_sha256'], cohorts['dev']+cohorts['probe'])
        timings['metric_gt_seconds'] = time.monotonic()-stage_started
        stage_started = time.monotonic()
        _progress(progress_path, 'trunk_loading', 0, 1)
        model = HeadStudyModel(arm, seed, size, head_spec=resolved_spec).to(device)
        load_trunk_artifact(model, trunk_artifact, receipt=receipt, synthetic=synthetic)
        frozen_digest = _trunk_digest(model)
        initial_head_digest = _head_parameter_digest(model)
        timings['model_and_trunk_seconds'] = time.monotonic()-stage_started
        _progress(progress_path, 'trunk_verified', 1, 1)
        extended_identity = refinement or resolved_spec is not None
        bindings = {'protocol_sha256': protocol_sha256, 'purpose': purpose, 'arm': arm, 'seed': seed,
                    'trunk_sha256': recipe['trunk_sha256'], 'trunk_digest': frozen_digest,
                    'manifest_file_sha256': manifest_file_sha, 'cache_hashes': {n: readers[n].metadata['metadata_sha256'] for n in names},
                    'target_epochs': target, 'synthetic': synthetic, 'recipe': recipe}
        if extended_identity:
            bindings.update(head_spec=resolved_spec, head_spec_sha256=head_spec_digest(resolved_spec))
        numeric = {'trunk': 'FP32', 'Q': HeadStudyModel.quantization, 'training_head': 'BF16' if device != 'cpu' else 'FP32',
                   'metric_head': 'FP32', 'effective_tf32': False, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'ema': False,
                   'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                   'cudnn_benchmark': torch.backends.cudnn.benchmark,
                   'cudnn_deterministic': torch.backends.cudnn.deterministic,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        hardware = {'device_uuid': 'cpu' if device == 'cpu' else recipe['gpu_for_seed'][str(seed)], 'torch_version': str(torch.__version__),
                    'cpu_threads': torch.get_num_threads()}
        optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01, momentum=.9)
        start = 0; history = []; selected_epoch = None; best = None; criterion = None
        if continuing:
            state = resume_training_state(resume_path)
            history = state['history']; selected_epoch = state['selected_epoch']; best = state['selected_dev_ap']
            start = len(history)
            if start > target or state.get('step') != start*len(readers['few'].metadata['fixed_layout']['blocks']) or state.get('total_steps') != target*len(readers['few'].metadata['fixed_layout']['blocks']):
                raise ValueError('resume progress differs from fixed target/layout')
            if [row.get('epoch') for row in history] != list(range(1, start+1)):
                raise ValueError('resume epoch history is not contiguous')
            measured = [row for row in history if 'dev' in row]
            chosen = max(measured, key=lambda row: row['dev']['AP50:95']) if measured else None
            if selected_epoch != (chosen['epoch'] if chosen else None) or best != (chosen['dev']['AP50:95'] if chosen else None):
                raise ValueError('resume selection disagrees with dev curve')
            if selected_epoch is not None:
                from .runtime import _has_symlink_component
                if set(state.get('selected_hashes', {})) != {f'selected-e{selected_epoch}.json', f'selected-e{selected_epoch}.pt'}:
                    raise ValueError('resume selected deployment hashes missing')
                for name, sha in state['selected_hashes'].items():
                    path = resume_path.parent/name
                    if Path(name).name != name or _has_symlink_component(path) or _sha(path) != sha:
                        raise ValueError('resume selected deployment hash mismatch')
            expected_resume = dict(bindings, step=start*len(readers['few'].metadata['fixed_layout']['blocks']),
                                   total_steps=target*len(readers['few'].metadata['fixed_layout']['blocks']))
            if load_resume(model, optimizer, resume_path, expected_state=expected_resume) != start:
                raise ValueError('resume curve does not cover completed epochs')
            if start == target:
                # The authoritative checkpoint may have committed the final epoch
                # before its presentation record. Only finish that publication; do
                # not optimize, evaluate, or alter the checkpoint's selection.
                if not previous_record.is_file():
                    raise ValueError('completed checkpoint lacks presentation baseline')
                record = _read_json(previous_record)
                expected_record = {'schema_version': 1, 'purpose': purpose, 'arm': arm, 'seed': seed,
                                   'protocol_sha256': protocol_sha256, 'target_epochs': target}
                if extended_identity:
                    expected_record.update(recipe=recipe, head_spec=resolved_spec,
                                           head_spec_sha256=head_spec_digest(resolved_spec))
                if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected_record.items()):
                    raise ValueError('completed checkpoint presentation baseline binding mismatch')
                initial_digest = record.get('initial_head_parameter_sha256')
                if not isinstance(initial_digest, str) or len(initial_digest) != 64 or any(c not in '0123456789abcdef' for c in initial_digest):
                    raise ValueError('completed checkpoint presentation baseline lacks initial head digest')
                last_digest = _head_parameter_digest(model)
                rng_before_last_validation = _rng_state()
                try:
                    last_model = load_checkpoint(resume_path.parent/'last.pt', resume_path.parent/'last.json',
                                                 trunk_artifact=trunk_artifact, synthetic_trunk=synthetic)
                finally:
                    # Construction calls manual_seed on all devices; restore the
                    # same complete RNG payload used by epoch-boundary resumes.
                    _restore_rng(rng_before_last_validation)
                last_state = last_model.detect.state_dict()
                resume_state = model.detect.state_dict()
                if set(last_state) != set(resume_state) or any(
                        not torch.equal(last_state[name].detach().cpu(), resume_state[name].detach().cpu()) for name in last_state):
                    raise ValueError('last deployment checkpoint disagrees with authoritative resume state')
                if _head_parameter_digest(last_model) != last_digest:
                    raise ValueError('last deployment digest disagrees with authoritative resume state')
                final_assets = ['resume.pt', 'last.pt', 'last.json'] + list(state['selected_hashes'])
                final_asset_hashes = {name: _sha(resume_path.parent/name) for name in final_assets}
                if resume_from is not None:
                    import shutil
                    output.mkdir(parents=True, exist_ok=False)
                    for name in final_assets:
                        shutil.copyfile(resume_path.parent/name, output/name)
                    if any(_sha(output/name) != sha for name, sha in final_asset_hashes.items()):
                        raise ValueError('copied final checkpoint assets failed hash verification')
                record.update({'schema_version': 1, 'purpose': purpose, 'synthetic': synthetic, 'arm': arm, 'seed': seed,
                               'protocol_sha256': protocol_sha256, 'target_epochs': target, 'epochs_completed': target,
                               'selected_epoch': selected_epoch, 'selection_metric': 'dev.AP50:95',
                               'selected_dev_ap': best,
                               'selected_config': f'selected-e{selected_epoch}.json' if selected_epoch is not None else None,
                               'last_config': 'last.json', 'history': history, 'hardware': hardware,
                               'numeric_policy': numeric, 'frozen_trunk_digest': frozen_digest,
                               'stage_timings': timings, 'last_head_parameter_sha256': last_digest})
                if extended_identity:
                    record.update(recipe=recipe, head_spec=resolved_spec,
                                  head_spec_sha256=head_spec_digest(resolved_spec))
                record['head_updated'] = record.get('last_head_parameter_sha256') != record.get('initial_head_parameter_sha256')
                _atomic_json(output/'training.json', record)
                return record
            if resume_from is not None:
                import shutil
                if selected_epoch is not None:
                    config_name = f'selected-e{selected_epoch}.json'
                    if not (resume_path.parent/config_name).is_file() or not (resume_path.parent/Path(config_name).with_suffix('.pt')).is_file():
                        raise ValueError('resume selected deployment files missing')
                output.mkdir(parents=True, exist_ok=False)
                if selected_epoch is not None:
                    for suffix in ('.pt', '.json'):
                        name = f'selected-e{selected_epoch}{suffix}'
                        shutil.copyfile(resume_path.parent/name, output/name)
        else: output.mkdir(parents=True, exist_ok=False)
        batches = cache_batches(readers['few'], seed=seed, ids=cohorts['few'], batch_size=batch)
        end = target if stop_epoch is None else stop_epoch
        if not start < end <= target: raise ValueError('invalid interruption boundary')
        for epoch in range(start, end):
            started = time.monotonic()
            result = train(model, batches, epochs=target, seed=seed, start_epoch=epoch, stop_epoch=epoch+1,
                           optimizer=optimizer, criterion=criterion, device=device, restored_rng=continuing,
                           batch_progress=lambda row: _progress(progress_path, 'train_batch', row['step'], target*batches.steps))
            criterion = result['criterion']
            row = result['history'][0]; row['train_seconds'] = time.monotonic()-started
            if _trunk_digest(model) != frozen_digest: raise RuntimeError('frozen trunk weights/BN changed')
            if purpose == 'main' and ((epoch+1) % cadence == 0 or epoch+1 == target):
                measured = time.monotonic()
                row['dev'] = evaluate_cache(model, readers['dev'], gt, device=device, progress_path=progress_path)
                row['probe'] = evaluate_cache(model, readers['probe'], gt, device=device, progress_path=progress_path)
                row['gap'] = row['probe']['AP50:95']-row['dev']['AP50:95']
                row['evaluation_seconds'] = time.monotonic()-measured
                ap = row['dev']['AP50:95']
                if best is None or ap > best:
                    best = ap; selected_epoch = epoch+1
                    save_checkpoint(model, output/f'selected-e{selected_epoch}.pt', output/f'selected-e{selected_epoch}.json', receipt)
            row['cache_read_seconds'] = readers['few'].read_seconds
            row['cache_read_bytes'] = readers['few'].read_bytes
            if device != 'cpu':
                row['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
                row['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            history.append(row)
            save_checkpoint(model, output/'last.pt', output/'last.json', receipt)
            # The resume file commits progress last and atomically; it is the sole
            # authority after an interrupted epoch, not the presentation JSON.
            selected_hashes = {} if selected_epoch is None else {f'selected-e{selected_epoch}{suffix}':
                               _sha(output/f'selected-e{selected_epoch}{suffix}') for suffix in ('.json', '.pt')}
            save_training_state(output/'resume.pt', result, epoch=epoch+1, **bindings,
                                history=history, selected_epoch=selected_epoch, selected_dev_ap=best, selected_hashes=selected_hashes)
            record = {'schema_version': 1, 'purpose': purpose, 'synthetic': synthetic, 'arm': arm, 'seed': seed,
                      'protocol_sha256': protocol_sha256, 'target_epochs': target, 'epochs_completed': epoch+1,
                      'selected_epoch': selected_epoch, 'selection_metric': 'dev.AP50:95', 'selected_dev_ap': best,
                      'selected_config': f'selected-e{selected_epoch}.json' if selected_epoch is not None else None, 'last_config': 'last.json',
                      'history': history, 'hardware': hardware, 'numeric_policy': numeric, 'frozen_trunk_digest': frozen_digest,
                      'stage_timings': timings, 'initial_head_parameter_sha256': initial_head_digest,
                      'last_head_parameter_sha256': _head_parameter_digest(model)}
            if extended_identity:
                record.update(recipe=recipe, head_spec=resolved_spec,
                              head_spec_sha256=head_spec_digest(resolved_spec))
            record['head_updated'] = record['last_head_parameter_sha256'] != initial_head_digest
            _atomic_json(output/'training.json', record)
            _progress(progress_path, 'epoch_committed', epoch+1, target)
        return record


def main(argv=None):
    import os
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('pilot', 'run'):
        s = sub.add_parser(name)
        for key in ('device', 'arm', 'cache', 'output', 'manifest', 'receipt', 'trunk-artifact'):
            s.add_argument('--'+key, required=True)
        s.add_argument('--seed', type=int, required=True)
        s.add_argument('--owned-root'); s.add_argument('--ledger'); s.add_argument('--reservation')
        s.add_argument('--synthetic', action='store_true')
        s.add_argument('--resume-from', help='authoritative epoch-boundary checkpoint; output remains a fresh supervised job tree')
        if name == 'pilot':
            s.add_argument('--imgsz', type=int, required=True); s.add_argument('--epochs', type=int, required=True)
            s.add_argument('--batch-size', type=int, required=True); s.add_argument('--producer', required=True)
            s.add_argument('--head-spec', help='optional canonical spec JSON for the seed-997 technical pilot')
        else:
            s.add_argument('--protocol', required=True); s.add_argument('--dev-cache', required=True)
            s.add_argument('--probe-cache', required=True); s.add_argument('--annotations', required=True)
    a = p.parse_args(argv)
    if a.arm not in ALL_ARMS: p.error('unknown arm')
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from .runtime import AllocationLedger, _has_symlink_component, canonical_gpu_uuid
    from .cache import _read_json, _sha, _progress
    progress_path = None
    if a.device == 'cpu':
        if not a.synthetic: raise RuntimeError('CPU is reserved for explicit synthetic fixtures')
        owned_root = Path(os.environ['HEAD_SHARING_TEST_ROOT']).resolve(strict=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; device = 'cpu'
        job_root = owned_root
    else:
        if a.synthetic: raise RuntimeError('synthetic worker is CPU-only')
        if not all((a.owned_root, a.ledger, a.reservation)): p.error('CUDA requires owned root, ledger and reservation')
        owned_root = Path(a.owned_root)
        if (_has_symlink_component(owned_root) or not owned_root.is_dir() or owned_root.absolute() != owned_root.resolve(strict=True)
                or not (owned_root/'STUDY_OWNER.txt').is_file() or (owned_root/'STUDY_OWNER.txt').is_symlink()):
            raise RuntimeError('canonical marker-owned root required')
        ledger_path = Path(a.ledger)
        if _has_symlink_component(ledger_path) or not ledger_path.resolve(strict=True).is_relative_to(owned_root):
            raise RuntimeError('ledger escaped owned root')
        if any(_has_symlink_component(ledger_path/name) for name in ('LEDGER_OWNER', 'ledger.json')):
            raise RuntimeError('linked ledger marker/state')
        ledger = AllocationLedger(ledger_path)
        from .supervisor import child_entry
        if canonical_gpu_uuid(a.device) != a.device: raise RuntimeError('device must be canonical UUID')
        child_entry(ledger, a.reservation, a.device)
        job = ledger.get_job(a.reservation); job_root = Path(job['cwd']); progress_path = Path(job['progress_file'])
        if (_has_symlink_component(job_root) or not job_root.resolve(strict=True).is_relative_to(owned_root)
                or _has_symlink_component(progress_path) or progress_path.parent.resolve(strict=True) != job_root.resolve(strict=True)):
            raise RuntimeError('supervisor job/progress confinement failed')
        gate = Path(os.environ['HEAD_SHARING_LAUNCH_GATE'])
        if _has_symlink_component(gate) or gate.absolute() != job_root/'.supervisor-ready':
            raise RuntimeError('launch gate escaped supervisor job')
        yolo_child = Path(os.environ.get('YOLO_CONFIG_DIR', ''))/'Ultralytics'
        if _has_symlink_component(yolo_child) or not yolo_child.is_dir():
            raise RuntimeError('native config child is missing or linked')
        for key in ('YOLO_CONFIG_DIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'TMPDIR', 'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR',
                    'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
            path = Path(os.environ.get(key, ''))
            if _has_symlink_component(path) or not path.resolve(strict=True).is_relative_to(job_root):
                raise RuntimeError('native config/cache escaped supervisor job')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != a.device: raise RuntimeError('assigned UUID mismatch')
        device = 'cuda'
    # Validate every declared path before importing the native runtime.
    for key in ('cache', 'dev_cache', 'probe_cache', 'manifest', 'receipt', 'trunk_artifact', 'annotations', 'protocol', 'resume_from', 'output', 'head_spec'):
        value = getattr(a, key, None)
        if value is None: continue
        path = Path(value).absolute()
        if _has_symlink_component(path) or not path.resolve(strict=False).is_relative_to(owned_root):
            raise RuntimeError('declared path must be link-free beneath owned root: '+key)
        if key == 'output' and not path.resolve(strict=False).is_relative_to(job_root):
            raise RuntimeError('output escaped supervisor job')
    if a.command == 'run':
        from .protocol import verify_frozen
        protocol = verify_frozen(a.protocol, synthetic=a.synthetic); recipe = protocol['recipe']; protocol_sha = protocol['recipe_sha256']
        if a.seed not in recipe['seed_list'] or a.arm not in recipe['arms']: raise ValueError('unassigned matrix member')
        if not a.synthetic and recipe['gpu_for_seed'].get(str(a.seed)) != a.device: raise ValueError('seed assigned to another physical GPU')
    else:
        if a.seed != 997: raise ValueError('technical pilot seed is 997 only')
        pilot_spec = None
        if a.head_spec:
            from .head_spec import canonicalize_head_spec
            pilot_spec = canonicalize_head_spec(_read_json(Path(a.head_spec)))
        metadata = _read_json(Path(a.cache)/'metadata.json')
        if not a.synthetic and (metadata['count'] != 512 or len(metadata['cohort_ids']['few']) != 512):
            raise ValueError('pilot requires its separate technical 512-image cache')
        recipe = {'imgsz': a.imgsz, 'epochs': a.epochs, 'batch_size': a.batch_size, 'cadence': a.epochs,
                  'producer': a.producer, 'split_sha256': _sha(Path(a.manifest)), 'trunk_sha256': metadata['trunk_sha256'],
                  'quantization': metadata['quantization'], 'cache_evidence': {'few': {'metadata_file_sha256': _sha(Path(a.cache)/'metadata.json')}},
                  'synthetic': a.synthetic, 'gpu_for_seed': {str(a.seed): a.device}}
        protocol_sha = None
    if a.command == 'run': pilot_spec = None
    _progress(progress_path, 'ownership_verified', 1, 1)
    import torch
    import cv2
    if device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError('one visible reserved GPU required')
        actual = canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
        ledger.validate_device(a.reservation, actual)
        if actual != a.device: raise RuntimeError('actual UUID mismatch')
    budget = max(1, min(3, int(os.environ.get('HEAD_SHARING_CPU_BUDGET', '1'))))
    torch.set_num_threads(budget); cv2.setNumThreads(budget)
    from .protocol import NUMERIC_POLICY
    # Match the actual frozen-feature producer. NVIDIA_TF32_OVERRIDE=0 disables
    # effective TF32; CUDA reductions are not claimed bitwise reproducible.
    torch.backends.cuda.matmul.allow_tf32 = NUMERIC_POLICY['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = NUMERIC_POLICY['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark = NUMERIC_POLICY['cudnn_benchmark']
    torch.backends.cudnn.deterministic = NUMERIC_POLICY['cudnn_deterministic']
    torch.use_deterministic_algorithms(NUMERIC_POLICY['deterministic_algorithms'])
    _progress(progress_path, 'native_import_complete', 1, 1)
    paths = {'few': a.cache}
    if a.command == 'run': paths.update(dev=a.dev_cache, probe=a.probe_cache)
    artifact = a.trunk_artifact
    if a.synthetic: artifact = torch.load(artifact, map_location='cpu', weights_only=True)
    experiment(purpose='pilot' if a.command == 'pilot' else 'main', arm=a.arm, seed=a.seed, recipe=recipe,
               protocol_sha256=protocol_sha, cache_paths=paths, manifest_path=a.manifest,
               annotations_path=getattr(a, 'annotations', None), trunk_artifact=artifact,
               receipt=_read_json(Path(a.receipt)), output=a.output, device=device, synthetic=a.synthetic,
               resume_from=a.resume_from, progress_path=progress_path, head_spec=pilot_spec)
    return 0

if __name__=="__main__": raise SystemExit(main())
