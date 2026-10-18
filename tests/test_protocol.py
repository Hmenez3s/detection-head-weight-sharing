import copy
import json
import pytest
from head_sharing.protocol import (REQUIRED_RECIPE, NUMERIC_POLICY, SELECTION, LOOPED_ARCHITECTURE,
                                   LOOPED_MATRIX, OLD_FREEZE_SHA256, OLD_PROTOCOL_FILE_SHA256,
                                   OLD_RECIPE_SHA256, freeze, verify_frozen, validate_recipe)
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


def test_looped_matrix_requires_fixed_architecture_and_native_reference():
    value=recipe(); value['arms']=list(LOOPED_MATRIX); value['environment']={
        'study_id':'looped_v1','looped_architecture':copy.deepcopy(LOOPED_ARCHITECTURE),
        'reference_native':{
            'protocol':{'path':'source/protocol.json','sha256':'c'*64,'recipe_sha256':'d'*64},
            'freeze':{'path':'source/freeze.json','sha256':'e'*64}}}
    assert validate_recipe(value,synthetic=True)['arms']==list(LOOPED_MATRIX)
    value['environment']['looped_architecture']['alpha']=0.2
    with pytest.raises(ValueError,match='architecture'):
        validate_recipe(value,synthetic=True)
    value['environment']['looped_architecture']=LOOPED_ARCHITECTURE
    value['environment']['study_id']='legacy'
    with pytest.raises(ValueError,match='architecture'):
        validate_recipe(value,synthetic=True)
    value=recipe(); value['arms']=list(LOOPED_MATRIX); value['environment']={
        'study_id':'looped_v1','looped_architecture':copy.deepcopy(LOOPED_ARCHITECTURE),
        'reference_native':{'protocol':{'path':'../escape.json','sha256':'c'*64,'recipe_sha256':'d'*64},
                            'freeze':{'path':'source/freeze.json','sha256':'e'*64}}}
    from head_sharing.evaluate import _portable_relative
    for unsafe in ('../escape.json', 'C:/escape.json', 'C:escape.json', 'source/file.json:stream'):
        value['environment']['reference_native']['protocol']['path'] = unsafe
        assert not _portable_relative(unsafe)
        with pytest.raises(ValueError,match='normalized'):
            validate_recipe(value,synthetic=True)


def test_production_looped_recipe_pins_hashes_not_basenames_and_exact_dose():
    value=recipe(); value.update(arms=list(LOOPED_MATRIX),synthetic=False,imgsz=512,batch_size=32,
                                 block_size=32,epochs=40,cadence=5,probe_size=1000,producer=GPU_UUIDS[1])
    value['gpu_for_seed']={str(seed):GPU_UUIDS[1 if seed in (11,33,55) else 0] for seed in (11,22,33,44,55)}
    value['cache_evidence']={name:{'metadata_file_sha256':'a'*64} for name in ('few','dev','probe')}
    value['environment']={'study_id':'looped_v1','looped_architecture':copy.deepcopy(LOOPED_ARCHITECTURE),
        'reference_native':{'protocol':{'path':'portable/source-protocol.bin','sha256':OLD_PROTOCOL_FILE_SHA256,
                                       'recipe_sha256':OLD_RECIPE_SHA256},
                            'freeze':{'path':'portable/source-freeze.bin','sha256':OLD_FREEZE_SHA256}}}
    assert validate_recipe(value,synthetic=False)['arms']==list(LOOPED_MATRIX)
    value['batch_size']=16; value['block_size']=16
    with pytest.raises(ValueError,match='dose'):
        validate_recipe(value,synthetic=False)


def refinement_recipe(*, synthetic=True, epochs=2):
    value = recipe()
    value.update(seed_list=[11,33,55], arms=["loop_ab2"], imgsz=64 if synthetic else 512,
                 batch_size=2 if synthetic else 32, block_size=2 if synthetic else 32,
                 epochs=epochs, cadence=1 if synthetic else 5, probe_size=2 if synthetic else 1000,
                 synthetic=synthetic, producer="cpu" if synthetic else GPU_UUIDS[1])
    value["gpu_for_seed"] = {str(seed): GPU_UUIDS[1] for seed in (11,33,55)}
    value["cache_evidence"] = ({name: {"metadata_file_sha256": "a"*64} for name in ("few","dev","probe")}
                               if not synthetic else {"few":"x", "dev":"y", "probe":"z"})
    value["environment"] = {"study_id":"ab2_refinement_v1",
        "head_spec":{"version":1,"variant":"learned_alpha","class_inner":96},
        "reference_baseline":{"protocol":{"path":"baseline/source-protocol.json","sha256":"c"*64,
            "recipe_sha256":"11ad424d07404fd1687c897fb1db51ffc1287a331b2022b73854f14be752e3a2" if not synthetic else "d"*64},
            "freeze":{"path":"looped-20261017T181804Z/selection/selected.json" if not synthetic else "baseline/selected.json",
                "sha256":"df3d874822a1c8015c312ac6f0ce8c59b4d64c1494c6ca55f5ddd70c5d7c2a47" if not synthetic else "e"*64}}}
    return value


def test_refinement_recipe_accepts_canonical_specs_without_narrowing_model_schema():
    value = refinement_recipe()
    assert validate_recipe(value, synthetic=True)["arms"] == ["loop_ab2"]
    from head_sharing.head_spec import canonicalize_head_spec
    specs = [None] + [canonicalize_head_spec({"version":1,"variant":variant,"class_inner":width})
                      for variant in ("learned_alpha", "step_norm", "scale_gain", "dw5")
                      for width in (96, 48)]
    for spec in specs:
        candidate = refinement_recipe()
        candidate["environment"]["head_spec"] = spec
        assert validate_recipe(candidate, synthetic=True)["environment"]["head_spec"] == spec
    for mutate in (
        lambda x: x.update(seed_list=[11,33,55,77]),
        lambda x: x.update(arms=["native"]),
        lambda x: x["environment"].update(head_spec={"version":1,"variant":"baseline","class_inner":96}),
        lambda x: x["environment"].update(head_spec={"version":1,"variant":"unknown","class_inner":96}),
        lambda x: x["environment"].pop("head_spec"),
        lambda x: x["environment"]["reference_baseline"]["freeze"].update(path="../selected.json"),
        lambda x: x["environment"]["reference_baseline"]["protocol"].update(sha256="not-a-digest"),
    ):
        invalid = refinement_recipe(); mutate(invalid)
        with pytest.raises(ValueError): validate_recipe(invalid, synthetic=True)
    production = refinement_recipe(synthetic=False, epochs=80)
    assert validate_recipe(production, synthetic=False)["epochs"] == 80
    for dose in (40,80):
        production = refinement_recipe(synthetic=False, epochs=dose)
        production["environment"]["reference_baseline"]["protocol"]["recipe_sha256"] = "0"*64
        with pytest.raises(ValueError, match="pinned"):
            validate_recipe(production, synthetic=False)
    production = refinement_recipe(synthetic=False, epochs=40)
    production["environment"]["reference_baseline"]["protocol"]["recipe_sha256"] = "11ad424d07404fd1687c897fb1db51ffc1287a331b2022b73854f14be752e3a2"
    assert validate_recipe(production, synthetic=False)["epochs"] == 40
    for dose in (40,80):
        production = refinement_recipe(synthetic=False, epochs=dose)
        production["environment"]["reference_baseline"]["protocol"]["recipe_sha256"] = "11ad424d07404fd1687c897fb1db51ffc1287a331b2022b73854f14be752e3a2"
        for spec in ([None] + [{"version":1,"variant":variant,"class_inner":width}
                               for variant in ("learned_alpha", "step_norm", "scale_gain", "dw5")
                               for width in (96,48)]):
            production["environment"]["head_spec"] = spec
            assert validate_recipe(production, synthetic=False)["environment"]["head_spec"] == spec
    production["epochs"] = 81
    with pytest.raises(ValueError, match="dose"):
        validate_recipe(production, synthetic=False)


def test_tiny_geometry_is_synthetic_only_and_real_evidence_is_typed():
    value=recipe();value['imgsz']=64;value['probe_size']=2
    assert validate_recipe(value,synthetic=True)['imgsz']==64
    value=recipe();value['synthetic']=False;value['producer']=GPU_UUIDS[1]
    with pytest.raises(ValueError,match='metadata_file_sha256'):
        validate_recipe(value,synthetic=False)
    value['cache_evidence']={cohort:{'metadata_file_sha256':'c'*64} for cohort in ('few','dev','probe')}
    assert validate_recipe(value,synthetic=False)['probe_size']==1000
