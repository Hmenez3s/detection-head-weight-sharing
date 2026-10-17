"""Freeze selected checkpoints before final inference; score immutable predictions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ARMS = ("native", "independent_k2", "shared_k2")
from . import LOOPED_ARMS
LOOPED_MATRIX = ("native", *LOOPED_ARMS)
SEEDS = (11, 22, 33, 44, 55)
POLICY = {
    "quantization": "fp32_fp16_fp32_v1", "conf_threshold": .001,
    "max_det": 300, "metric_head_dtype": "float32",
    "NVIDIA_TF32_OVERRIDE": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
    "cudnn_benchmark": False, "cudnn_deterministic": False,
    "deterministic_algorithms": False,
}


def _sha(path: Path, progress_path: str | Path | None = None) -> str:
    h = hashlib.sha256(); count = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block); count += len(block)
            if progress_path is not None:
                from .cache import _progress
                _progress(progress_path, 'hash_bytes', count, path.stat().st_size)
    return h.hexdigest()


def _json(path: Path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.evaluation-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _relative(path: Path, root: Path) -> str:
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("evidence path escapes owned root")
    return path.relative_to(root).as_posix()


def _portable_relative(value: object) -> bool:
    from pathlib import PurePosixPath
    return (isinstance(value,str) and bool(value) and "\\" not in value and ":" not in value and
            str(PurePosixPath(value)) == value and not PurePosixPath(value).is_absolute() and
            all(part not in ("", ".", "..") for part in value.split("/")))


def _validated_run(run: Path, root: Path, recipe: dict, protocol_sha: str) -> dict:
    """Validate a complete run against its own source recipe and protocol identity."""
    from .checkpoint import resume_training_state
    from .runtime import _has_symlink_component

    run = Path(run)
    if _has_symlink_component(run) or not run.is_dir() or not run.resolve(strict=True).is_relative_to(root):
        raise ValueError("run directory missing, linked, or outside owned root")
    record = _json(run / 'training.json')
    arm, seed = record.get('arm'), record.get('seed')
    if (record.get('schema_version') != 1 or record.get('purpose') != 'main' or
            record.get('synthetic') is not recipe['synthetic'] or
            record.get('protocol_sha256') != protocol_sha or
            record.get('target_epochs') != recipe['epochs'] or
            record.get('epochs_completed') != recipe['epochs']):
        raise ValueError("run is incomplete or bound to a different frozen protocol")
    state = resume_training_state(run / 'resume.pt')
    if (state.get('purpose') != 'main' or state.get('protocol_sha256') != protocol_sha or
            state.get('arm') != arm or state.get('seed') != seed or
            state.get('target_epochs') != recipe['epochs'] or state.get('recipe') != recipe):
        raise ValueError("authoritative resume binding mismatch")
    history = state.get('history')
    if (not isinstance(history, list) or len(history) != recipe['epochs'] or
            [row.get('epoch') for row in history] != list(range(1, recipe['epochs'] + 1))):
        raise ValueError("resume history does not cover contiguous fixed target")
    schedule = [epoch for epoch in range(1, recipe['epochs'] + 1)
                if epoch % recipe['cadence'] == 0 or epoch == recipe['epochs']]
    measured = [row for row in history if 'dev' in row]
    if [row['epoch'] for row in measured] != schedule:
        raise ValueError("resume dev evaluations differ from fixed cadence")
    aps = [row.get('dev', {}).get('AP50:95') for row in measured]
    if any(type(ap) not in (int, float) or not math.isfinite(ap) for ap in aps):
        raise ValueError("scheduled dev selection history is invalid")
    best_index = max(range(len(aps)), key=lambda i: aps[i])
    selected_epoch = measured[best_index]['epoch']
    if (state.get('selected_epoch') != selected_epoch or state.get('selected_dev_ap') != aps[best_index] or
            type(state.get('step')) is not int or state.get('step') < recipe['epochs'] or
            state.get('step') != state.get('total_steps')):
        raise ValueError("authoritative resume selection/progress mismatch")
    for field, expected in {'purpose':'main', 'protocol_sha256':protocol_sha, 'arm':arm, 'seed':seed,
                            'target_epochs':recipe['epochs'], 'synthetic':recipe['synthetic'], 'recipe':recipe}.items():
        if state.get(field) != expected: raise ValueError("authoritative resume binding mismatch")
    expected_files = {f'selected-e{selected_epoch}.json', f'selected-e{selected_epoch}.pt'}
    hashes = state.get('selected_hashes')
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise ValueError("selected checkpoint file binding missing")
    for name, digest in hashes.items():
        path = run / name
        if _has_symlink_component(path) or _sha(path) != digest:
            raise ValueError("selected checkpoint digest mismatch")
    config_name = record.get('selected_config')
    if config_name != f'selected-e{selected_epoch}.json':
        raise ValueError("training record selected config disagrees with resume state")
    config = _json(run / config_name)
    if (config.get('seed') != seed or config.get('arm') != arm or config.get('imgsz') != recipe['imgsz'] or
            config.get('quantization') != recipe['quantization'] or
            config.get('trunk_receipt', {}).get('trunk_artifact_sha256') != recipe['trunk_sha256']):
        raise ValueError("selected deployment config/trunk identity mismatch")
    presentation = record.get('history')
    if (record.get('selected_epoch') != selected_epoch or record.get('selected_dev_ap') != aps[best_index] or
            record.get('selection_metric') != 'dev.AP50:95' or
            record.get('frozen_trunk_digest') != state.get('trunk_digest') or record.get('selected_config') != config_name or
            not isinstance(presentation, list) or
            [r.get('epoch') for r in presentation] != list(range(1, recipe['epochs'] + 1)) or
            [(r.get('epoch'), r.get('dev', {}).get('AP50:95')) for r in presentation if 'dev' in r] !=
            [(r['epoch'], r['dev']['AP50:95']) for r in measured]):
        raise ValueError("training presentation record disagrees with authoritative resume")
    return {'seed':seed, 'arm':arm, 'selected_epoch':selected_epoch, 'selected_dev_ap':aps[best_index],
            'config':_relative(run / config_name, root), 'weights':_relative(run / f'selected-e{selected_epoch}.pt', root),
            'config_sha256':hashes[config_name], 'weights_sha256':hashes[f'selected-e{selected_epoch}.pt'],
            'run_record_sha256':_sha(run / 'training.json'), 'resume_sha256':_sha(run / 'resume.pt'),
            'trunk_sha256':recipe['trunk_sha256']}


def _verified_member_files(row: dict, root: Path, recipe: dict, protocol_sha: str) -> dict:
    """Recheck frozen byte identities without deserializing any model/resume.

    Authoritative tensor-state selection is checked by freeze_selected once;
    subsequent verification binds its exact bytes, retaining the lazy scoring
    and pre-inference ownership boundary.
    """
    from .runtime import _has_symlink_component
    paths = {}
    hashes = {}
    for field, hash_field in (('config','config_sha256'),('weights','weights_sha256')):
        ref = row.get(field)
        if not _portable_relative(ref):
            raise ValueError('freeze contains nonportable checkpoint reference')
        target = root / ref
        if (_has_symlink_component(target) or not target.is_file() or
                not target.resolve(strict=True).is_relative_to(root)):
            raise ValueError('frozen checkpoint escaped its owned root')
        paths[field] = target
        hashes[hash_field] = _sha(target)
        if hashes[hash_field] != row.get(hash_field):
            raise ValueError('selected checkpoint digest mismatch')
    run = paths['config'].parent
    for name, hash_field in (('training.json','run_record_sha256'),('resume.pt','resume_sha256')):
        target = run / name
        if _has_symlink_component(target) or not target.is_file():
            raise ValueError('frozen producer record or resume missing or linked')
        hashes[hash_field] = _sha(target)
        if hashes[hash_field] != row.get(hash_field):
            raise ValueError('frozen producer record or resume digest mismatch')
    config, record = _json(paths['config']), _json(run/'training.json')
    seed, arm, epoch = row.get('seed'), row.get('arm'), row.get('selected_epoch')
    if (type(epoch) is not int or epoch not in range(1,recipe['epochs']+1) or
            row.get('trunk_sha256') != recipe['trunk_sha256'] or
            config.get('seed') != seed or config.get('arm') != arm or config.get('imgsz') != recipe['imgsz'] or
            config.get('quantization') != recipe['quantization'] or
            config.get('trunk_receipt',{}).get('trunk_artifact_sha256') != recipe['trunk_sha256']):
        raise ValueError('frozen member config/selection/trunk identity mismatch')
    if (record.get('schema_version') != 1 or record.get('purpose') != 'main' or
            record.get('seed') != seed or record.get('arm') != arm or
            record.get('synthetic') is not recipe['synthetic'] or
            record.get('protocol_sha256') != protocol_sha or
            record.get('target_epochs') != recipe['epochs'] or record.get('epochs_completed') != recipe['epochs'] or
            record.get('selected_epoch') != epoch or record.get('selected_dev_ap') != row.get('selected_dev_ap') or
            record.get('selection_metric') != 'dev.AP50:95' or
            record.get('selected_config') != f'selected-e{epoch}.json'):
        raise ValueError('frozen training record/source binding mismatch')
    ap = row.get('selected_dev_ap')
    if type(ap) not in (int,float) or not math.isfinite(ap):
        raise ValueError('frozen member selection score missing')
    return {'seed':seed,'arm':arm,'selected_epoch':epoch,'selected_dev_ap':ap,
            'config':_relative(paths['config'],root),'weights':_relative(paths['weights'],root),
            **hashes,'trunk_sha256':recipe['trunk_sha256']}


def _rooted_reference(root: Path, binding: dict, *, expected_sha: str | None = None) -> Path:
    from .runtime import _has_symlink_component
    ref = binding['path']
    path = root / ref
    if (_has_symlink_component(path) or not path.is_file() or not path.resolve(strict=True).is_relative_to(root) or
            _sha(path) != binding['sha256'] or (expected_sha is not None and binding['sha256'] != expected_sha)):
        raise ValueError("referenced source artifact is missing, changed, or outside owned root")
    return path


def _projection(recipe: dict) -> dict:
    keys = ('seed_list','imgsz','batch_size','block_size','epochs','cadence','probe_size','optimizer','gains',
            'quantization','producer','trunk_sha256','split_sha256','cache_evidence','gpu_for_seed','selection','numeric_policy')
    return {key: recipe[key] for key in keys}


def _source_bundle_root(root: Path, freeze_path: Path, protocol_path: Path, protocol_ref: str) -> Path:
    from .runtime import _has_symlink_component
    if not _portable_relative(protocol_ref):
        raise ValueError("source freeze protocol reference is not portable")
    ref = Path(protocol_ref)
    for candidate in (freeze_path.parent, *freeze_path.parents):
        if candidate.resolve(strict=True).is_relative_to(root) and not _has_symlink_component(candidate):
            target = candidate / ref
            if target.is_file() and target.resolve(strict=True) == protocol_path.resolve(strict=True):
                return candidate.resolve(strict=True)
    raise ValueError("source freeze protocol cannot be resolved in its owned bundle")


def freeze_selected(protocol_path: str | Path, run_dirs: list[str | Path], output: str | Path,
                    owned_root: str | Path) -> dict:
    """Freeze new complete runs and verified native reference members atomically."""
    from .protocol import verify_frozen, OLD_FREEZE_SHA256, OLD_PROTOCOL_FILE_SHA256, OLD_RECIPE_SHA256
    from .runtime import _has_symlink_component

    root = Path(owned_root)
    if (_has_symlink_component(root) or not root.is_dir() or root.absolute() != root.resolve(strict=True) or
            not (root / 'STUDY_OWNER.txt').is_file() or (root / 'STUDY_OWNER.txt').is_symlink()):
        raise ValueError("canonical marker-owned root required")
    root = root.resolve(strict=True)
    protocol_path = Path(protocol_path)
    if _has_symlink_component(protocol_path) or not protocol_path.resolve(strict=True).is_relative_to(root):
        raise ValueError("protocol must be a link-free reference inside owned root")
    protocol = verify_frozen(protocol_path, synthetic=bool(_json(protocol_path)['recipe']['synthetic']))
    recipe, protocol_sha = protocol['recipe'], protocol['recipe_sha256']
    matrix = recipe.get('arms')
    if recipe.get('seed_list') != list(SEEDS) or matrix not in (list(ARMS), list(LOOPED_MATRIX)):
        raise ValueError("protocol differs from fixed evaluation matrix")
    looped = matrix == list(LOOPED_MATRIX)
    expected_arms = tuple(LOOPED_ARMS) if looped else ARMS
    if not recipe['synthetic'] and (recipe.get('epochs') != 40 or recipe.get('cadence') != 5):
        raise ValueError("protocol differs from fixed evaluation schedule")
    members = {}
    if looped:
        environment = recipe['environment']; references = environment['reference_native']
        source_protocol_path = _rooted_reference(root, references['protocol'], expected_sha=None if recipe['synthetic'] else OLD_PROTOCOL_FILE_SHA256)
        source_protocol = verify_frozen(source_protocol_path, synthetic=recipe['synthetic'])
        if (source_protocol['recipe_sha256'] != references['protocol']['recipe_sha256'] or
                (not recipe['synthetic'] and source_protocol['recipe_sha256'] != OLD_RECIPE_SHA256) or
                _projection(source_protocol['recipe']) != _projection(recipe)):
            raise ValueError("new recipe inherited projection differs from pinned native source")
        source_freeze_path = _rooted_reference(root, references['freeze'], expected_sha=None if recipe['synthetic'] else OLD_FREEZE_SHA256)
        source_freeze = _json(source_freeze_path)
        if (source_freeze.get('schema_version') != 1 or source_freeze.get('protocol_sha256') != source_protocol['recipe_sha256']):
            raise ValueError("native source freeze header differs from pinned source protocol")
        source_root = _source_bundle_root(root,source_freeze_path,source_protocol_path,source_freeze.get('protocol',''))
        if (source_freeze.get('protocol') != _relative(source_protocol_path,source_root) or
                source_freeze.get('trunk_sha256') != source_protocol['recipe']['trunk_sha256'] or
                source_freeze.get('split_sha256') != source_protocol['recipe']['split_sha256'] or
                source_freeze.get('imgsz') != source_protocol['recipe']['imgsz'] or
                source_freeze.get('batch_size') != source_protocol['recipe']['batch_size'] or
                source_freeze.get('quantization') != source_protocol['recipe']['quantization'] or
                source_freeze.get('producer') != source_protocol['recipe']['producer'] or
                source_freeze.get('synthetic') is not source_protocol['recipe']['synthetic'] or
                source_freeze.get('selection') != source_protocol['recipe']['selection'] or
                source_freeze.get('policy') != POLICY or not isinstance(source_freeze.get('members'),list) or len(source_freeze['members']) != 15):
            raise ValueError("native source freeze header differs from pinned recipe")
        old_rows = {(row.get('seed'), row.get('arm')): row for row in source_freeze['members']
                    if isinstance(row, dict) and row.get('arm') == 'native'}
        if set(old_rows) != {(seed, 'native') for seed in SEEDS}:
            raise ValueError("native source freeze does not contain exactly the five native controls")
        for seed in SEEDS:
            old = old_rows[(seed, 'native')]
            checked = _validated_run((source_root / old['config']).parent, source_root,
                                     source_protocol['recipe'], source_protocol['recipe_sha256'])
            for key in ('selected_epoch','selected_dev_ap','config','weights','config_sha256','weights_sha256',
                        'run_record_sha256','resume_sha256','trunk_sha256'):
                if checked[key] != old.get(key): raise ValueError("native member differs from pinned source freeze")
            checked['config'] = _relative(source_root / old['config'], root)
            checked['weights'] = _relative(source_root / old['weights'], root)
            checked['source'] = {'kind':'reused_native','protocol_sha256':source_protocol['recipe_sha256'],
                                 'freeze_sha256':references['freeze']['sha256']}
            members[(seed,'native')] = checked
    if len(run_dirs) != len(SEEDS) * len(expected_arms):
        raise ValueError("freeze requires exactly ten new runs or fifteen legacy run directories")
    for run_value in run_dirs:
        checked = _validated_run(Path(run_value), root, recipe, protocol_sha)
        arm, seed = checked['arm'], checked['seed']; key = (seed,arm)
        if key in members or seed not in SEEDS or arm not in expected_arms:
            raise ValueError("duplicate or unexpected run in fixed matrix")
        if looped: checked['source'] = {'kind':'current','protocol_sha256':protocol_sha}
        members[key] = checked
    expected = {(seed,arm) for seed in SEEDS for arm in (('native', *expected_arms) if looped else expected_arms)}
    if set(members) != expected: raise ValueError("freeze matrix has missing seed/arm members")
    result = {'schema_version':2 if looped else 1,'protocol_sha256':protocol_sha,
              'protocol':_relative(protocol_path,root),'trunk_sha256':recipe['trunk_sha256'],
              'split_sha256':recipe['split_sha256'],'imgsz':recipe['imgsz'],'batch_size':recipe['batch_size'],
              'quantization':recipe['quantization'],'producer':recipe['producer'],'synthetic':recipe['synthetic'],
              'selection':recipe['selection'],'policy':POLICY,
              'members':[members[(seed,arm)] for seed in SEEDS for arm in (('native', *expected_arms) if looped else expected_arms)]}
    if looped: result['reference_native'] = recipe['environment']['reference_native']
    output = Path(output)
    if (_has_symlink_component(output) or output.exists() or
            not output.absolute().parent.resolve(strict=False).is_relative_to(root)):
        raise FileExistsError("freeze output must be fresh and inside owned root")
    _atomic_json(output,result)
    return result


def member_source_binding(frozen: dict, member: dict, current_freeze_file_sha: str) -> dict:
    """Return the immutable source identity for one row without importing ML code."""
    source = member.get('source')
    if frozen.get('schema_version') == 1:
        if source is not None: raise ValueError("legacy member cannot declare mixed provenance")
        return {'protocol_sha256':frozen['protocol_sha256'],'freeze_sha256':current_freeze_file_sha}
    if frozen.get('schema_version') != 2 or not isinstance(source, dict):
        raise ValueError("invalid mixed-provenance member source")
    if source.get('kind') == 'current' and set(source) == {'kind','protocol_sha256'}:
        if source['protocol_sha256'] != frozen['protocol_sha256']:
            raise ValueError("current member protocol binding mismatch")
        return {'protocol_sha256':source['protocol_sha256'],'freeze_sha256':current_freeze_file_sha}
    if source.get('kind') == 'reused_native' and set(source) == {'kind','protocol_sha256','freeze_sha256'}:
        reference = frozen.get('reference_native', {})
        if (source['protocol_sha256'] != reference.get('protocol', {}).get('recipe_sha256') or
                source['freeze_sha256'] != reference.get('freeze', {}).get('sha256')):
            raise ValueError("reused native member source binding mismatch")
        return {'protocol_sha256':source['protocol_sha256'],'freeze_sha256':source['freeze_sha256']}
    raise ValueError("invalid member source binding")


def verify_freeze(path: str | Path, owned_root: str | Path) -> dict:
    """Recheck every immutable selected artifact before any holdout input is opened."""
    from .runtime import _has_symlink_component
    from .protocol import verify_frozen, OLD_FREEZE_SHA256, OLD_PROTOCOL_FILE_SHA256, OLD_RECIPE_SHA256
    root_path = Path(owned_root)
    if (_has_symlink_component(root_path) or not root_path.is_dir() or root_path.absolute() != root_path.resolve(strict=True) or
            not (root_path / 'STUDY_OWNER.txt').is_file() or (root_path / 'STUDY_OWNER.txt').is_symlink()):
        raise ValueError("canonical marker-owned root required")
    root = root_path.resolve(strict=True); freeze_path = Path(path)
    if _has_symlink_component(freeze_path) or not freeze_path.resolve(strict=True).is_relative_to(root):
        raise ValueError("freeze must be a link-free reference inside owned root")
    value = _json(freeze_path); version = value.get('schema_version')
    if version not in (1,2) or value.get('policy') != POLICY:
        raise ValueError("freeze schema or inference policy mismatch")
    rows = value.get('members')
    if not isinstance(rows,list) or len(rows) != 15:
        raise ValueError("freeze is not the complete fifteen-member matrix")
    protocol_ref = value.get('protocol')
    protocol_path = root / protocol_ref if isinstance(protocol_ref,str) else root
    if (not _portable_relative(protocol_ref) or _has_symlink_component(protocol_path) or not protocol_path.is_file()):
        raise ValueError("freeze protocol reference is missing, linked, or nonportable")
    protocol = verify_frozen(protocol_path, synthetic=value.get('synthetic') is True); recipe = protocol['recipe']
    looped = recipe.get('arms') == list(LOOPED_MATRIX)
    if (version != (2 if looped else 1) or protocol['recipe_sha256'] != value.get('protocol_sha256') or
            recipe.get('seed_list') != list(SEEDS) or recipe.get('arms') not in (list(ARMS),list(LOOPED_MATRIX)) or
            value.get('trunk_sha256') != recipe['trunk_sha256'] or value.get('split_sha256') != recipe['split_sha256'] or
            value.get('imgsz') != recipe['imgsz'] or value.get('batch_size') != recipe['batch_size'] or
            value.get('quantization') != recipe['quantization'] or value.get('producer') != recipe['producer'] or
            value.get('selection') != recipe['selection'] or value.get('synthetic') is not recipe['synthetic']):
        raise ValueError("freeze identity differs from immutable protocol")
    if looped:
        references = recipe['environment']['reference_native']
        if value.get('reference_native') != references: raise ValueError("freeze native reference differs from recipe")
        source_protocol_path = _rooted_reference(root,references['protocol'],expected_sha=None if recipe['synthetic'] else OLD_PROTOCOL_FILE_SHA256)
        source_protocol = verify_frozen(source_protocol_path,synthetic=recipe['synthetic'])
        source_freeze_path = _rooted_reference(root,references['freeze'],expected_sha=None if recipe['synthetic'] else OLD_FREEZE_SHA256)
        source_freeze = _json(source_freeze_path)
        if (source_protocol['recipe_sha256'] != references['protocol']['recipe_sha256'] or
                (not recipe['synthetic'] and source_protocol['recipe_sha256'] != OLD_RECIPE_SHA256) or
                source_freeze.get('schema_version') != 1 or source_freeze.get('protocol_sha256') != source_protocol['recipe_sha256']):
            raise ValueError("referenced native source header is invalid")
        source_root = _source_bundle_root(root,source_freeze_path,source_protocol_path,source_freeze.get('protocol',''))
        if (_projection(source_protocol['recipe']) != _projection(recipe) or
                source_freeze.get('protocol') != _relative(source_protocol_path,source_root) or
                source_freeze.get('trunk_sha256') != source_protocol['recipe']['trunk_sha256'] or
                source_freeze.get('split_sha256') != source_protocol['recipe']['split_sha256'] or
                source_freeze.get('imgsz') != source_protocol['recipe']['imgsz'] or
                source_freeze.get('batch_size') != source_protocol['recipe']['batch_size'] or
                source_freeze.get('quantization') != source_protocol['recipe']['quantization'] or
                source_freeze.get('producer') != source_protocol['recipe']['producer'] or
                source_freeze.get('synthetic') is not source_protocol['recipe']['synthetic'] or
                source_freeze.get('selection') != source_protocol['recipe']['selection'] or
                source_freeze.get('policy') != POLICY or not isinstance(source_freeze.get('members'),list) or len(source_freeze['members']) != 15):
            raise ValueError("new recipe/source freeze projection differs from pinned native source")
        old_rows = {(r.get('seed'),r.get('arm')):r for r in source_freeze['members'] if isinstance(r,dict) and r.get('arm')=='native'}
        if set(old_rows) != {(seed,'native') for seed in SEEDS}: raise ValueError("source freeze native matrix is incomplete")
    else:
        references = None; source_protocol = source_freeze = None; old_rows = {}
    current_sha = _sha(freeze_path); accepted_arms = tuple(recipe['arms']); seen = set()
    expected_keys = {(s,a) for s in SEEDS for a in (('native',*LOOPED_ARMS) if looped else accepted_arms)}
    for row in rows:
        if not isinstance(row,dict): raise ValueError("invalid freeze member")
        key=(row.get('seed'),row.get('arm'))
        if key in seen or key not in expected_keys: raise ValueError("invalid freeze member matrix")
        seen.add(key)
        is_reused = looped and row['arm'] == 'native'
        binding = member_source_binding(value,row,current_sha)
        if is_reused:
            if key not in old_rows or row.get('source',{}).get('kind') != 'reused_native': raise ValueError("native source row missing")
            old = old_rows[key]
            if (any(row.get(k) != old.get(k) for k in ('selected_epoch','selected_dev_ap','config_sha256','weights_sha256','run_record_sha256','resume_sha256','trunk_sha256')) or
                    row.get('config') != _relative(source_root / old['config'],root) or
                    row.get('weights') != _relative(source_root / old['weights'],root)):
                raise ValueError("native row differs from original frozen source")
            checked = _verified_member_files(old,source_root,source_protocol['recipe'],source_protocol['recipe_sha256'])
            if any(checked[k] != old.get(k) for k in ('config','weights','config_sha256','weights_sha256','run_record_sha256','resume_sha256','selected_epoch','selected_dev_ap','trunk_sha256')):
                raise ValueError("native source artifacts failed complete run validation")
            if binding != {'protocol_sha256':source_protocol['recipe_sha256'],'freeze_sha256':references['freeze']['sha256']}:
                raise ValueError("native source identity changed")
            continue
        if version == 1 and 'source' in row: raise ValueError("legacy freeze row schema changed")
        if version == 2 and (row.get('source') != {'kind':'current','protocol_sha256':protocol['recipe_sha256']}):
            raise ValueError("new member source identity changed")
        if (type(row.get('selected_epoch')) is not int or row['selected_epoch'] not in range(1,recipe['epochs']+1) or
                row.get('trunk_sha256') != recipe['trunk_sha256']): raise ValueError("invalid frozen member selection/trunk identity")
        for field,hash_field in (('config','config_sha256'),('weights','weights_sha256')):
            ref=row.get(field)
            if not _portable_relative(ref):
                raise ValueError("freeze contains nonportable checkpoint reference")
            target=root/ref
            if _has_symlink_component(target) or not target.is_file() or not target.resolve(strict=True).is_relative_to(root) or _sha(target)!=row.get(hash_field):
                raise ValueError("frozen checkpoint changed")
        checked=_verified_member_files(row,root,recipe,protocol['recipe_sha256'])
        if any(checked.get(k)!=row.get(k) for k in checked): raise ValueError("frozen run record/resume binding changed")
        if member_source_binding(value,row,current_sha) != {'protocol_sha256':protocol['recipe_sha256'],'freeze_sha256':current_sha}:
            raise ValueError("current member binding mismatch")
    if seen != expected_keys: raise ValueError("freeze matrix has missing members")
    return value


def predict(freeze_path: str | Path, *, seed: int, arm: str, owned_root: str | Path, annotations_path: str | Path,
            image_root: str | Path, image_ids_path: str | Path, trunk_artifact: str | Path,
            receipt_path: str | Path, output: str | Path, device: str = 'cpu', synthetic: bool = False,
            progress_path: str | Path | None = None) -> dict:
    """Run common-Q native-head prediction only after the complete set re-verifies."""
    # This ordering is deliberate: no holdout path is opened before every frozen
    # selected checkpoint has been re-hashed and its matrix verified.
    frozen = verify_freeze(freeze_path, owned_root)
    if frozen['synthetic'] is not synthetic:
        raise ValueError("synthetic mode must match the frozen protocol")
    if device == 'cpu' and not synthetic:
        raise ValueError("CPU final inference is reserved for explicit synthetic fixtures")
    if seed not in SEEDS or arm not in (*ARMS, *LOOPED_ARMS):
        raise ValueError("prediction requires one fixed seed/arm member")
    member = next((row for row in frozen['members'] if (row['seed'], row['arm']) == (seed, arm)), None)
    if member is None: raise ValueError("requested member is absent from complete freeze")
    if frozen['schema_version'] == 2 and member.get('source', {}).get('kind') == 'reused_native':
        raise ValueError("reused native members must retain their original prediction identity")
    root = Path(owned_root).resolve(strict=True)
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from .runtime import _has_symlink_component
    job_root = None
    if device != 'cpu':
        from .runtime import AllocationLedger, canonical_gpu_uuid
        from .supervisor import child_entry
        token = os.environ.get('HEAD_SHARING_RESERVATION')
        gate = os.environ.get('HEAD_SHARING_LAUNCH_GATE')
        ledger_path = Path(os.environ.get('HEAD_SHARING_LEDGER', ''))
        if (not token or not gate or not ledger_path.is_dir() or _has_symlink_component(ledger_path) or
                canonical_gpu_uuid(device) != device):
            raise RuntimeError("CUDA prediction requires a canonical supervised reservation")
        from .protocol import verify_frozen
        recipe = verify_frozen(root / frozen['protocol'])['recipe']
        if recipe['gpu_for_seed'][str(seed)] != device:
            raise RuntimeError("inference UUID differs from frozen seed assignment")
        ledger = AllocationLedger(ledger_path)
        child_entry(ledger, token, device)
        job = ledger.get_job(token)
        job_root = Path(job['cwd'])
        progress_path = Path(job['progress_file'])
        if (job.get('uuid') != device or _has_symlink_component(job_root) or
                not job_root.resolve(strict=True).is_relative_to(root) or
                _has_symlink_component(progress_path) or
                progress_path.parent.resolve(strict=True) != job_root.resolve(strict=True) or
                not progress_path.is_file()):
            raise RuntimeError("supervisor job/progress confinement failed")
    source_binding = member_source_binding(frozen, member, _sha(Path(freeze_path)))
    output = Path(output)
    if (_has_symlink_component(output) or output.exists() or
            not output.absolute().parent.resolve(strict=False).is_relative_to(root) or
            (job_root is not None and not output.absolute().parent.resolve(strict=False).is_relative_to(job_root.resolve(strict=True)))):
        raise FileExistsError("prediction output must be fresh inside its owned job root")
    import cv2
    import numpy as np
    import torch
    threads = max(1, min(3, int(os.environ.get('HEAD_SHARING_CPU_BUDGET', '3'))))
    torch.set_num_threads(threads)
    cv2.setNumThreads(0)  # Serial preprocessing also works on GCD builds, which reject positive limits.
    execution_uuid = 'cpu'; torch_device = 'cpu'
    if device != 'cpu':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("reserved inference requires exactly one visible CUDA device")
        execution_uuid = canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
        ledger.validate_device(token, execution_uuid, cuda_module=torch.cuda)
        torch_device = 'cuda:0'
    from .checkpoint import load_checkpoint
    from .geometry import letterbox, decoded_rows, COCO80_TO_91
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    annotations_path, image_ids_path = Path(annotations_path), Path(image_ids_path)
    annotations_sha = _sha(annotations_path, progress_path=progress_path)
    annotations = _json(annotations_path)
    ids_sha = _sha(image_ids_path, progress_path=progress_path)
    ids_value = json.loads(image_ids_path.read_text(encoding='utf-8'))
    ids = ids_value.get('image_ids') if isinstance(ids_value, dict) else ids_value
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(type(i) is not int for i in ids):
        raise ValueError("holdout image IDs must be a nonempty unique integer list")
    images = {row['id']: row for row in annotations.get('images', [])}
    if set(ids) - images.keys():
        raise ValueError("holdout image metadata does not cover requested IDs")
    image_root = Path(image_root).resolve(strict=True)
    receipt = _json(Path(receipt_path))
    if receipt.get('trunk_artifact_sha256') != frozen['trunk_sha256']:
        raise ValueError("trunk receipt differs from freeze")
    trunk = Path(trunk_artifact)
    if not trunk.is_absolute(): trunk = root / trunk
    if not synthetic and _sha(trunk, progress_path=progress_path) != frozen['trunk_sha256']:
        raise ValueError("trunk artifact differs from freeze")
    if output.exists() or output.is_symlink():
        raise FileExistsError("prediction output is immutable")
    from .cache import _progress
    image_rows = []
    _progress(progress_path, 'image_hashes', 0, len(ids))
    for image_id in ids:
        row = images[image_id]
        path = (image_root / row['file_name']).resolve(strict=True)
        if not path.is_relative_to(image_root) or path.is_symlink(): raise ValueError("holdout image escaped root")
        image_rows.append((image_id, row, path, _sha(path, progress_path=progress_path)))
        _progress(progress_path, 'image_hashes', len(image_rows), len(ids))
    config = root / member['config']; weights = root / member['weights']
    trunk_source = torch.load(trunk, map_location='cpu', weights_only=True) if synthetic else trunk
    model = load_checkpoint(weights, config, trunk_artifact=trunk_source,
                            synthetic_trunk=synthetic, map_location='cpu').to(torch_device).float().eval()
    all_predictions = []
    batch_size = frozen['batch_size']
    with torch.no_grad(), torch.autocast(device_type=torch.device('cpu' if device == 'cpu' else torch_device).type, enabled=False):
        for start in range(0, len(image_rows), batch_size):
            chunk = image_rows[start:start + batch_size]
            tensors, transforms = [], []
            for image_id, row, path, _ in chunk:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError("cannot decode holdout image")
                if [bgr.shape[1], bgr.shape[0]] != [row['width'], row['height']]: raise ValueError("holdout dimensions mismatch")
                meta = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), frozen['imgsz'])
                transforms.append(meta)
                tensors.append(np.ascontiguousarray(meta.image.transpose(2, 0, 1)))
            pixels = torch.from_numpy(np.stack(tensors)).to(device=('cpu' if device == 'cpu' else torch_device), dtype=torch.float32) / 255.
            decoded = model.forward_images(pixels)[0]
            for output_rows, (image_id, _, _, _), meta in zip(decoded, chunk, transforms, strict=True):
                all_predictions.extend(decoded_rows(output_rows, image_id, COCO80_TO_91, .001, 300, meta))
            _progress(progress_path, 'prediction_batches', min(start + len(chunk), len(image_rows)), len(image_rows))
    del model
    image_set_sha = hashlib.sha256(json.dumps([(i, digest) for i, _, _, digest in image_rows], separators=(',', ':')).encode()).hexdigest()
    content_sha = hashlib.sha256(json.dumps(all_predictions, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    payload = {'schema_version': 1, 'seed': seed, 'arm': arm, 'selected_epoch': member['selected_epoch'],
               'execution_uuid': execution_uuid,
               'config_sha256': member['config_sha256'], 'weights_sha256': member['weights_sha256'],
               'freeze_sha256': source_binding['freeze_sha256'], 'protocol_sha256': source_binding['protocol_sha256'],
               'image_ids': ids, 'image_ids_sha256': ids_sha, 'image_set_sha256': image_set_sha,
               'annotations_sha256': annotations_sha, 'prediction_policy': POLICY,
               'predictions': all_predictions, 'content_sha256': content_sha}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents completed predictions from ever being replaced.
    with output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
            zipped.write(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())
        raw.flush(); os.fsync(raw.fileno())
    receipt_out = output.with_suffix(output.suffix + '.json')
    receipt_value = {'schema_version': 1, 'prediction_sha256': _sha(output),
                     'execution_uuid': execution_uuid,
                     'seed': seed, 'arm': arm, 'selected_epoch': member['selected_epoch'],
                     'config_sha256': member['config_sha256'], 'weights_sha256': member['weights_sha256'],
                     'freeze_sha256': payload['freeze_sha256'], 'protocol_sha256': payload['protocol_sha256'],
                     'image_ids_sha256': ids_sha, 'image_set_sha256': image_set_sha,
                     'annotations_sha256': annotations_sha, 'content_sha256': content_sha,
                     'policy': POLICY, 'image_ids': ids}
    _atomic_json(receipt_out, receipt_value)
    return {'seed': seed, 'arm': arm, 'selected_epoch': member['selected_epoch'],
            'prediction_sha256': receipt_value['prediction_sha256'], 'count': len(all_predictions), 'image_ids': ids}


def score_prediction(freeze_path: str | Path, *, owned_root: str | Path,
                     annotations_path: str | Path, predictions_path: str | Path) -> dict:
    """Replay one immutable member's predictions without importing inference/runtime code."""
    frozen = verify_freeze(freeze_path, owned_root)
    from .runtime import _has_symlink_component
    root = Path(owned_root).resolve(strict=True)
    pred_path = Path(predictions_path)
    receipt_path = pred_path.with_suffix(pred_path.suffix + '.json')
    if (_has_symlink_component(pred_path) or _has_symlink_component(receipt_path) or
            not pred_path.resolve(strict=True).is_relative_to(root) or
            not receipt_path.resolve(strict=True).is_relative_to(root)):
        raise ValueError("prediction artifacts must be link-free inside owned root")
    receipt = _json(receipt_path)
    freeze_sha = _sha(Path(freeze_path))
    annotation_sha = _sha(Path(annotations_path))
    if (_sha(pred_path) != receipt.get('prediction_sha256') or receipt.get('policy') != POLICY or
            receipt.get('annotations_sha256') != annotation_sha):
        raise ValueError("prediction file, annotation, or inference policy differs from immutable receipt")
    with gzip.open(pred_path, 'rt', encoding='utf-8') as stream:
        payload = json.load(stream)
    identity_fields = ('seed', 'arm', 'selected_epoch', 'execution_uuid', 'config_sha256', 'weights_sha256',
                       'freeze_sha256', 'protocol_sha256', 'image_ids_sha256',
                       'image_set_sha256', 'annotations_sha256', 'content_sha256')
    if any(payload.get(field) != receipt.get(field) for field in identity_fields):
        raise ValueError("prediction receipt identity binding mismatch")
    member = next((m for m in frozen['members'] if (m['seed'], m['arm']) ==
                   (payload.get('seed'), payload.get('arm'))), None)
    if member is None: raise ValueError("prediction member is absent from complete freeze")
    expected_binding = member_source_binding(frozen, member, freeze_sha)
    if (payload.get('selected_epoch') != member['selected_epoch'] or
            payload.get('config_sha256') != member['config_sha256'] or
            payload.get('weights_sha256') != member['weights_sha256'] or
            payload.get('protocol_sha256') != expected_binding['protocol_sha256'] or
            payload.get('freeze_sha256') != expected_binding['freeze_sha256'] or
            receipt.get('protocol_sha256') != expected_binding['protocol_sha256'] or
            receipt.get('freeze_sha256') != expected_binding['freeze_sha256'] or
            payload.get('prediction_policy') != POLICY):
        raise ValueError("prediction member/source differs from selected frozen checkpoint")
    from .protocol import verify_frozen
    recipe = verify_frozen(root / frozen['protocol'], synthetic=frozen['synthetic'])['recipe']
    expected_uuid = 'cpu' if frozen['synthetic'] else recipe['gpu_for_seed'][str(member['seed'])]
    if payload.get('execution_uuid') != expected_uuid:
        raise ValueError("prediction execution UUID differs from frozen seed assignment")
    ids = payload.get('image_ids')
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or
            any(type(i) is not int for i in ids) or receipt.get('image_ids') != ids):
        raise ValueError("prediction image identity is invalid")
    if hashlib.sha256(json.dumps(payload.get('predictions'), sort_keys=True, separators=(',', ':')).encode()).hexdigest() != payload.get('content_sha256'):
        raise ValueError("prediction content digest mismatch")
    from .coco import score as coco_score
    scores = coco_score(_json(Path(annotations_path)), ids, payload['predictions'])
    return dict(scores, seed=payload['seed'], arm=payload['arm'], selected_epoch=payload['selected_epoch'],
                execution_uuid=payload['execution_uuid'], prediction_sha256=receipt['prediction_sha256'],
                freeze_sha256=expected_binding['freeze_sha256'], protocol_sha256=expected_binding['protocol_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze-selected')
    freeze.add_argument('--protocol', required=True); freeze.add_argument('--owned-root', required=True)
    freeze.add_argument('--output', required=True); freeze.add_argument('--run', action='append', required=True)
    infer = sub.add_parser('predict')
    for key in ('freeze', 'owned-root', 'annotations', 'image-root', 'image-ids', 'trunk-artifact', 'receipt', 'output'):
        infer.add_argument('--'+key, required=True)
    infer.add_argument('--seed', type=int, required=True); infer.add_argument('--arm', choices=(*ARMS, *LOOPED_ARMS), required=True)
    infer.add_argument('--device', default='cpu'); infer.add_argument('--synthetic', action='store_true')
    score = sub.add_parser('score')
    score.add_argument('--freeze', required=True); score.add_argument('--owned-root', required=True)
    score.add_argument('--annotations', required=True); score.add_argument('--predictions', required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze-selected':
        result = freeze_selected(args.protocol, args.run, args.output, args.owned_root)
        print(json.dumps({'protocol_sha256': result['protocol_sha256'], 'members': len(result['members'])}, sort_keys=True))
        return 0
    if args.command == 'predict':
        result = predict(args.freeze, seed=args.seed, arm=args.arm, owned_root=args.owned_root, annotations_path=args.annotations,
                         image_root=args.image_root, image_ids_path=args.image_ids,
                         trunk_artifact=args.trunk_artifact, receipt_path=args.receipt,
                         output=args.output, device=args.device, synthetic=args.synthetic)
        print(json.dumps(result, sort_keys=True)); return 0
    scores = score_prediction(args.freeze, owned_root=args.owned_root,
                              annotations_path=args.annotations, predictions_path=args.predictions)
    print(json.dumps(scores, sort_keys=True)); return 0

if __name__ == '__main__':
    raise SystemExit(main())
