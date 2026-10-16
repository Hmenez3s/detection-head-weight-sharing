import random

import numpy as np
import torch

from head_sharing.checkpoint import load_resume, save_resume
from head_sharing.model import HeadStudyModel
from head_sharing.train import optimizer_groups


def test_resume_restores_python_numpy_and_torch_rng(tmp_path):
    torch.set_num_threads(1)
    model = HeadStudyModel("native", 11, 64)
    optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01, momentum=.9)
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    path = tmp_path / "resume.pt"
    save_resume(model, optimizer, path, epoch=0, training_state={"total_epochs": 2, "seed": 11})
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    restored = HeadStudyModel("native", 11, 64)
    restored_optimizer = torch.optim.SGD(optimizer_groups(restored, .0005), lr=.01, momentum=.9)
    assert load_resume(restored, restored_optimizer, path, expected_state={"total_epochs": 2, "seed": 11}) == 0
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_epoch_zero_restored_rng_is_not_reseeded(tmp_path, monkeypatch):
    import head_sharing.train as module
    model = HeadStudyModel('native', 11, 64)
    optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01, momentum=.9)
    path = tmp_path/'resume.pt'; save_resume(model, optimizer, path, epoch=0)
    restored = HeadStudyModel('native', 11, 64)
    restored_optimizer = torch.optim.SGD(optimizer_groups(restored, .0005), lr=.01, momentum=.9)
    assert load_resume(restored, restored_optimizer, path) == 0
    before = torch.get_rng_state().clone()
    def forbidden(seed): raise AssertionError('restored RNG was reseeded')
    monkeypatch.setattr(module, '_seed', forbidden)
    def batches(epoch): raise AssertionError('zero-length oracle should not consume batches')
    module.train(restored, batches, epochs=2, seed=11, optimizer=restored_optimizer, stop_epoch=0, restored_rng=True)
    assert torch.equal(torch.get_rng_state(), before)


def test_resume_rejects_native_objects_and_malformed_bindings_before_applying(tmp_path):
    import pytest
    from head_sharing.train import _native_criterion, save_training_state
    torch.set_num_threads(1)
    source = HeadStudyModel('shared_k2', 11, 64)
    optimizer = torch.optim.SGD(optimizer_groups(source, .0005), lr=.01, momentum=.9)
    criterion = _native_criterion(source, 2)
    path = tmp_path/'resume.pt'
    with pytest.raises(ValueError, match='primitives'):
        save_resume(source, optimizer, path, epoch=1, training_state={'criterion': criterion})
    assert not path.exists()
    result = {'model': source, 'optimizer': optimizer, 'criterion': criterion, 'seed': 11, 'step': 1, 'total_steps': 2}
    binding = {'arm': 'shared_k2', 'target_epochs': 2, 'protocol_sha256': 'frozen', 'trunk_sha256': 'trunk', 'cache_hashes': {'few': 'cache'}}
    save_training_state(path, result, epoch=1, **binding)
    payload = torch.load(path, weights_only=True)
    assert 'criterion' not in payload['training_state']
    target = HeadStudyModel('shared_k2', 22, 64)
    target_optimizer = torch.optim.SGD(optimizer_groups(target, .0005), lr=.01, momentum=.9)
    before = {k: v.clone() for k, v in target.detect.state_dict().items()}
    for key in binding:
        with pytest.raises(ValueError, match='binding'):
            load_resume(target, target_optimizer, path, expected_state=dict(binding, **{key: 'wrong'}))
        assert all(torch.equal(before[k], v) for k, v in target.detect.state_dict().items())
    for field, invalid in (('epoch', 3), ('rng', {'python': [], 'numpy': [], 'torch_cpu': torch.zeros(2)}),
                           ('optimizer', {'state': {}, 'param_groups': []})):
        malformed = dict(payload); malformed[field] = invalid
        invalid_path = tmp_path/(field+'.pt'); torch.save(malformed, invalid_path)
        with pytest.raises(ValueError): load_resume(target, target_optimizer, invalid_path, expected_state=binding)
        assert all(torch.equal(before[k], v) for k, v in target.detect.state_dict().items())


def test_atomic_resume_failure_preserves_completed_boundary(tmp_path, monkeypatch):
    import pytest
    import head_sharing.checkpoint as module
    model = HeadStudyModel('native', 11, 64)
    optimizer = torch.optim.SGD(optimizer_groups(model, .0005), lr=.01, momentum=.9)
    path = tmp_path/'resume.pt'
    save_resume(model, optimizer, path, epoch=0, training_state={'target_epochs': 2})
    before = path.read_bytes()
    def fail(*args, **kwargs): raise OSError('simulated interrupted serialize')
    monkeypatch.setattr(module.torch, 'save', fail)
    with pytest.raises(OSError): save_resume(model, optimizer, path, epoch=1, training_state={'target_epochs': 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
