import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import head_sharing.cache as cache
from head_sharing.model import HeadStudyModel

RECEIPT = {"trunk_artifact_sha256":"f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016", "trunk_tensors":468,"trunk_bytes":9109168,"head_index":23,"input_channels":[64,128,256],"source_variant":"U","source_seed":11,"source_epoch_zero_based":79,"source_training_epochs":80,"pre_exposure":"synthetic-only test fixture"}


def fixture(tmp_path):
    torch.set_num_threads(1)
    tmp_path.mkdir(parents=True,exist_ok=True)
    image_root=tmp_path/"images"; image_root.mkdir()
    rows=[]; coco_images=[]
    for i in range(6):
        p=image_root/f"{i}.png"
        assert cv2.imwrite(str(p),np.random.default_rng(i+50).integers(0,256,(32,40,3),dtype=np.uint8))
        rows.append({"image_id":i,"file_name":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        coco_images.append({"id":i,"file_name":p.name,"width":40,"height":32})
    cats=[{"category_id":i+1,"class_id":i,"name":f"c{i}"} for i in range(80)]
    ann={"images":coco_images,"categories":[{"id":x["category_id"],"name":x["name"]} for x in cats],"annotations":[{"id":i,"image_id":i%6,"category_id":1,"bbox":[2,3,8,9],"iscrowd":0} for i in range(6)]+[{"id":20,"image_id":1,"category_id":1,"bbox":[1,1,2,2],"iscrowd":1}]}
    ap=tmp_path/"ann.json"; ap.write_text(json.dumps(ann)); mp=tmp_path/"manifest.json"
    manifest={"ids":{"few":[0,1,2,3],"dev":[4,5]},"probe_ids":[0,2],"counts":{"few":4,"dev":2},"manifest_sha256":"a"*64,"source_annotation_sha256":hashlib.sha256(ap.read_bytes()).hexdigest(),"categories":cats,"images":rows}
    mp.write_text(json.dumps(manifest)); rp=tmp_path/"receipt.json"; rp.write_text(json.dumps(RECEIPT))
    model=HeadStudyModel(seed=123,imgsz=32).eval()
    state=model.detector.state_dict()
    artifact={"state_dict":{k:v.detach().cpu() for k,v in state.items() if k.startswith("model.") and int(k.split(".")[1])<23},"schema_version":1,"architecture":"yolo26n","upstream":"8.4.177","excluded_head_index":23}
    return image_root,ap,mp,rp,model,artifact


def build(tmp_path, cohort="few", batch_size=3):
    image_root,ap,mp,rp,model,artifact=fixture(tmp_path)
    out=tmp_path/f"cache-{cohort}"
    meta=cache.build_cache(image_root=image_root,annotations_path=ap,manifest_path=mp,receipt_path=rp,trunk_artifact=artifact,
        output=out,model=model,producer="cpu",cohort=cohort,batch_size=batch_size,synthetic=True)
    return image_root,out,meta,model


def test_cohorts_contiguous_views_tail_q_oracle_and_one_open_two_epochs(tmp_path,monkeypatch):
    image_root,out,meta,model=build(tmp_path)
    assert meta["cohort"]=="few" and set(meta["block_order"])=={"few"}
    assert len(meta["block_order"]["few"])==4 and meta["batch_size"]==3
    with cache.load_cache(out,expected_producer="cpu") as reader:
        assert all(a[0].flags.c_contiguous and a[1].flags.c_contiguous for a in reader.maps)
        ids=meta["block_order"]["few"]
        from head_sharing.data import epoch_batches
        planned=list(epoch_batches(meta["fixed_layout"],seed=7,epoch=0))
        assert all(tuple(group) in [tuple(x) for x in meta["fixed_layout"]["blocks"]] for group,_ in planned)
        orig=cache._sha; hashes=[]
        monkeypatch.setattr(cache,"_sha",lambda p:(hashes.append(p),orig(p))[1])
        epochs=[]
        for _ in range(2): epochs.append(list(cache.read_blocks(reader,"few",ids,batch_size=3)))
        assert len(hashes)==0
        assert [len(rows) for _,rows in epochs[0]]==[3,1]
        rgb=cv2.cvtColor(cv2.imread(str(image_root/"0.png")),cv2.COLOR_BGR2RGB)
        from head_sharing.geometry import letterbox
        for flipped in (False,True):
            x=torch.from_numpy(letterbox(rgb,32,flipped=flipped).image.transpose(2,0,1).copy()).float().unsqueeze(0)/255
            i=ids.index(0)
            disk=[torch.from_numpy(np.array(a[int(flipped),i:i+1],copy=True)).float() for a in reader.maps]
            expected=[v.to(torch.float16).float() for v in model.extract_features(x)]
            assert all(torch.equal(a,b) for a,b in zip(disk,expected,strict=True))
            assert cache.compare_cache(model,x,disk,atol=0,rtol=0)["decoded_max_abs"]==0
    monkeypatch.setattr(cache,"_sha",orig)
    _,devout,dev,_=build(tmp_path/"devcase","dev")
    assert dev["view_policy"]==["original"] and all(x["shape"][0]==1 for x in dev["feature_files"])
    _,probeout,probe,_=build(tmp_path/"probecase","probe")
    assert probe["cohort"]=="probe" and set(probe["block_order"]["probe"])=={0,2}
    with cache.load_cache(devout) as reader:
        assert [r["image_id"] for _,rows in cache.read_blocks(reader,"dev",dev["block_order"]["dev"],batch_size=32) for r in rows]==dev["block_order"]["dev"]
        with pytest.raises(ValueError,match="cohort"): list(cache.read_blocks(reader,"few",dev["block_order"]["dev"]))


def test_hash_before_annotation_parse_strict_trunk_and_metadata_corruption(tmp_path,monkeypatch):
    image_root,ap,mp,rp,model,artifact=fixture(tmp_path)
    out=tmp_path/"wrong-hash"
    manifest=json.loads(mp.read_text()); manifest["source_annotation_sha256"]="0"*64; mp.write_text(json.dumps(manifest))
    monkeypatch.setattr(cache,"_read_json",lambda p: (_ for _ in ()).throw(AssertionError("parsed")) if Path(p)==ap else json.loads(Path(p).read_text()))
    with pytest.raises(ValueError,match="annotation SHA256"):
        cache.build_cache(image_root=image_root,annotations_path=ap,manifest_path=mp,receipt_path=rp,trunk_artifact=artifact,output=out,model=model,producer="cpu",synthetic=True)
    assert not out.exists()
    monkeypatch.undo()
    with pytest.raises(ValueError,match="synthetic"):
        cache.build_cache(image_root=image_root,annotations_path=ap,manifest_path=mp,receipt_path=rp,trunk_artifact=artifact,output=out,model=model,producer="cpu")


@pytest.mark.parametrize("imgsz", [512, 640])
def test_native_quantized_disk_comparison_at_production_resolutions(tmp_path,imgsz):
    image_root,ap,mp,rp,model,artifact=fixture(tmp_path)
    model.imgsz=imgsz
    manifest=json.loads(mp.read_text()); manifest["ids"]={"few":[0],"dev":[4,5]}; manifest["counts"]={"few":1,"dev":2}; manifest["probe_ids"]=[0]
    mp.write_text(json.dumps(manifest))
    out=tmp_path/"bounded-cache"
    meta=cache.build_cache(image_root=image_root,annotations_path=ap,manifest_path=mp,receipt_path=rp,trunk_artifact=artifact,
        output=out,model=model,producer="cpu",cohort="few",batch_size=1,synthetic=True)
    from head_sharing.geometry import letterbox
    rgb=cv2.cvtColor(cv2.imread(str(image_root/"0.png")),cv2.COLOR_BGR2RGB)
    with cache.load_cache(out,expected_identity={"imgsz":imgsz,"cohort":"few"}) as reader:
        for flipped in (False,True):
            x=torch.from_numpy(letterbox(rgb,imgsz,flipped=flipped).image.transpose(2,0,1).copy()).float().unsqueeze(0)/255
            features=[torch.from_numpy(np.array(a[int(flipped),:1],copy=True)).float() for a in reader.maps]
            result=cache.compare_cache(model,x,features,atol=0,rtol=0)
            assert result["feature_max_abs"]==result["decoded_max_abs"]==0
            assert len(meta["fixed_layout"]["blocks"])==1


def test_wrong_identity_corrupt_shard_partial_and_noncontiguous_rejected(tmp_path):
    _,out,meta,_=build(tmp_path)
    with pytest.raises(ValueError,match="producer"): cache.load_cache(out,expected_producer="other")
    with pytest.raises(ValueError,match="identity"): cache.load_cache(out,expected_identity={"trunk_sha256":"bad"})
    with cache.load_cache(out) as reader:
        ids=reader.metadata["block_order"]["few"]
        with pytest.raises(ValueError,match="physical"): list(cache.read_blocks(reader,"few",ids[::-1]))
    metadata_path=out/"metadata.json"; tampered=json.loads(metadata_path.read_text()); tampered["view_policy"]=["feature_flip"]
    metadata_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match="identity"): cache.load_cache(out)
    tampered["view_policy"]=meta["view_policy"]; metadata_path.write_text(json.dumps(tampered))
    bad=out/meta["feature_files"][0]["file"]; bad.write_bytes(bad.read_bytes()[:-1])
    with pytest.raises(ValueError,match="hash"): cache.load_cache(out)
    partial=tmp_path/"partial"; partial.mkdir(); (partial/"metadata.json").write_text(json.dumps({"format":2,"complete":False}))
    with pytest.raises(ValueError,match="incomplete"): cache.load_cache(partial)


def test_original_targets_remain_paired_with_clean_features(tmp_path):
    _, _, metadata, _ = build(tmp_path)
    row = next(r for r in metadata['images'] if r['image_id'] == 0)
    # Original 40px-wide image: bbox [2, 3, 8, 9] has center x=6, not 34.
    assert row['targets_xywhn'][0][1] == pytest.approx(0.15)
    assert row['flipped']['targets_xywhn'][0][1] == pytest.approx(0.85)


def test_read_time_excludes_consumer_and_open_reports_hash_progress(tmp_path, monkeypatch):
    _, output, metadata, _ = build(tmp_path)
    progress = tmp_path / 'progress.jsonl'
    with cache.load_cache(output, progress_path=progress) as reader:
        records = [json.loads(line) for line in progress.read_text().splitlines()]
        assert len(records) >= 3 and all(r['phase'] == 'hashed_bytes' and r['done'] > 0 for r in records)
        clock = [0.0]
        monkeypatch.setattr(cache.time, 'monotonic', lambda: clock[0])
        for _ in cache.read_blocks(reader, 'few', metadata['block_order']['few'], batch_size=3):
            clock[0] += 10.0  # Consumer time must not be attributed to cache I/O.
        assert reader.read_seconds == 0.0


def test_heartbeat_visibility_does_not_flush_each_image_but_completion_is_durable(tmp_path, monkeypatch):
    from head_sharing import cache
    calls=[]
    monkeypatch.setattr(cache.os,'fsync',lambda fd:calls.append(fd))
    log=tmp_path/'progress.jsonl'
    cache._progress(log,'hashed_bytes',42,42)
    assert json.loads(log.read_text())['done']==42
    assert calls==[]
    cache._progress(log,'cache_complete',1,1)
    assert len(calls)==1
    cache._atomic_json(tmp_path/'metadata.json',{'complete':True})
    assert len(calls)==2


def test_each_blocking_feature_flush_has_visible_phase(monkeypatch, tmp_path):
    events = []
    original = np.memmap.flush
    def flush(array):
        events.append(('flush',))
        return original(array)
    monkeypatch.setattr(np.memmap, 'flush', flush)
    monkeypatch.setattr(cache, '_progress', lambda path, phase, done, total: events.append(('progress', phase, done, total)))
    build(tmp_path)
    locations = [i for i, event in enumerate(events) if event == ('flush',)]
    assert len(locations) == 3
    for level, position in enumerate(locations):
        assert events[position-1] == ('progress', 'flushing_features', level, 3)


def test_dev_comparison_uses_declared_dev_ids_not_disjoint_train_probe(tmp_path, monkeypatch, capsys):
    import head_sharing.checkpoint as checkpoint
    image_root, ap, mp, rp, model, artifact = fixture(tmp_path)
    model.imgsz = 512
    out = tmp_path/'dev512'
    metadata = cache.build_cache(image_root=image_root, annotations_path=ap, manifest_path=mp,
        receipt_path=rp, trunk_artifact=artifact, output=out, model=model, producer='cpu',
        cohort='dev', batch_size=3, synthetic=True)
    load = checkpoint.load_trunk_artifact
    monkeypatch.setattr(checkpoint, 'load_trunk_artifact',
        lambda target, path, receipt: load(target, artifact, receipt=receipt, synthetic=True))
    declared = metadata['block_order']['dev'][0]
    assert declared not in metadata['cohort_ids']['probe']
    args = ['compare', '--cache', str(out), '--expected-producer', 'cpu', '--image-root', str(image_root),
        '--receipt', str(rp), '--trunk-artifact', str(tmp_path/'synthetic-trunk'), '--imgsz', '512',
        '--device', 'cpu', '--atol', '0', '--rtol', '0', '--probe-id', str(declared)]
    assert cache.main(args) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert len(rows) == 1 and rows[0]['cohort'] == 'dev' and rows[0]['mode'] == 'matched_build_group'
    assert rows[0]['feature_max_abs'] == 0 and rows[0]['decoded_max_abs'] == 0
    args[-1] = str(metadata['cohort_ids']['probe'][0])
    with pytest.raises(ValueError, match='not declared for this cohort'):
        cache.main(args)
