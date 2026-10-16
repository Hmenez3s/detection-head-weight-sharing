import json
import pytest
from head_sharing.protocol import REQUIRED_RECIPE, NUMERIC_POLICY, SELECTION, freeze, verify_frozen, validate_recipe
from head_sharing.runtime import GPU_UUIDS


def recipe():
    return {"status":"FROZEN","seed_list":[11,22,33,44,55],"arms":["native","independent_k2","shared_k2"],"imgsz":512,"batch_size":2,"epochs":4,"cadence":2,"block_size":2,"probe_size":1000,"optimizer":{"name":"SGD","lr":.01,"momentum":.9,"weight_decay":.0005},"gains":{"box":7.5,"cls":.5,"dfl":1.5},"quantization":"fp32_fp16_fp32_v1","producer":"producer-gpu","environment":{"torch":"2.9.1"},"trunk_sha256":"a"*64,"split_sha256":"b"*64,"cache_evidence":{"few":"x","dev":"y","probe":"z"},"pilot_evidence":{"ids":[1]},"synthetic":True,"gpu_for_seed":{str(seed):GPU_UUIDS[int(seed in (11,33,55))] for seed in (11,22,33,44,55)},"selection":dict(SELECTION),"numeric_policy":dict(NUMERIC_POLICY)}


def test_immutable_synthetic_protocol_freeze(tmp_path):
    source=tmp_path/"recipe.json"; source.write_text(json.dumps(recipe()))
    path=tmp_path/"frozen.json"
    freeze(source,path,synthetic=True)
    assert verify_frozen(path,synthetic=True)["recipe"]["seed_list"]==[11,22,33,44,55]
    with pytest.raises(FileExistsError): freeze(source,path,synthetic=True)
    value=json.loads(path.read_text()); value["recipe"]["epochs"]=5; path.write_text(json.dumps(value))
    with pytest.raises(ValueError): verify_frozen(path,synthetic=True)


def test_freeze_rejects_incomplete_or_non_synthetic_recipe():
    with pytest.raises(ValueError): validate_recipe({k:v for k,v in recipe().items() if k!="pilot_evidence"},synthetic=True)
    with pytest.raises(ValueError): validate_recipe(recipe(),synthetic=False)


@pytest.mark.parametrize('change',[
    {'gpu_for_seed':{'11':GPU_UUIDS[0]}},
    {'selection':{'cohort':'holdout','metric':'AP50:95','tie':'earliest','training_length':'fixed_epochs'}},
    {'numeric_policy':{**NUMERIC_POLICY,'NVIDIA_TF32_OVERRIDE':'1'}},
    {'block_size':4},
    {'trunk_sha256':'not-an-artifact-digest'},
])
def test_fixed_triplets_selection_numerics_and_artifact_bindings(change):
    value=recipe();value.update(change)
    with pytest.raises(ValueError):validate_recipe(value,synthetic=True)


def test_tiny_geometry_is_synthetic_only_and_real_evidence_is_typed():
    value=recipe();value['imgsz']=64;value['probe_size']=2
    assert validate_recipe(value,synthetic=True)['imgsz']==64
    value=recipe();value['synthetic']=False;value['producer']=GPU_UUIDS[1]
    with pytest.raises(ValueError,match='metadata_file_sha256'):
        validate_recipe(value,synthetic=False)
    value['cache_evidence']={cohort:{'metadata_file_sha256':'c'*64} for cohort in ('few','dev','probe')}
    assert validate_recipe(value,synthetic=False)['probe_size']==1000
