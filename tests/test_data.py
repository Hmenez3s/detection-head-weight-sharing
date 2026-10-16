import hashlib
import json

import pytest

from head_sharing.data import compact_manifest, epoch_batches, fixed_blocks


def test_compact_manifest_and_fixed_paired_blocks(tmp_path):
    rows = []
    for i in range(6):
        b = bytes([i]); (tmp_path / f"{i}.jpg").write_bytes(b)
        rows.append({"file_name": f"{i}.jpg", "image_id": i, "sha256": hashlib.sha256(b).hexdigest()})
    source = {"ids": {"few": [0,1,2,3], "dev": [4], "full": list(range(6))},
              "counts": {"few": 4, "dev": 1, "full": 6}, "categories": [{"id": 1}],
              "images": rows, "manifest_sha256": "source", "source_annotation_sha256": "ann",
              "data_yaml": {"path": "/private"}}
    compact = compact_manifest(source, tmp_path, probe_ids=[0,1])
    assert "data_yaml" not in compact and "/private" not in json.dumps(compact)
    assert compact["probe_ids"] == [0,1]
    assert set(compact["ids"]) == set(compact["counts"]) == {"few", "dev"}
    assert compact_manifest(compact, tmp_path, probe_ids=compact["probe_ids"]) == compact
    a, b = fixed_blocks(list(range(6)), 2), fixed_blocks(list(range(6)), 2)
    assert a == b and a["tail_policy"] == "keep_partial"
    assert list(epoch_batches(a, 7, 2)) == list(epoch_batches(b, 7, 2))
    assert set(sum((x for x,_ in epoch_batches(a, 7, 2)), [])) == set(range(6))
    bad = dict(source); bad["ids"] = dict(source["ids"], dev=[0])
    with pytest.raises(ValueError, match="overlap"):
        compact_manifest(bad, tmp_path)
    bad = dict(source); bad["images"] = rows[:4]
    with pytest.raises(ValueError, match="missing image"):
        compact_manifest(bad, tmp_path)


def test_image_filename_cannot_escape_explicit_root(tmp_path):
    root = tmp_path / "images"; root.mkdir()
    outside = tmp_path / "outside.jpg"; outside.write_bytes(b"sentinel")
    source = {"ids": {"few": [1], "dev": [], "full": [1]},
              "images": [{"image_id": 1, "file_name": "../outside.jpg",
                          "sha256": hashlib.sha256(b"sentinel").hexdigest()}],
              "categories": [], "manifest_sha256": "m", "source_annotation_sha256": "a"}
    with pytest.raises(ValueError, match="relative to image root|escapes image root"):
        compact_manifest(source, root, probe_ids=[1])
    source["images"][0]["file_name"] = str(outside)
    with pytest.raises(ValueError, match="relative to image root"):
        compact_manifest(source, root, probe_ids=[1])
    link = root / "escape"; link.symlink_to(tmp_path, target_is_directory=True)
    source["images"][0]["file_name"] = "escape/outside.jpg"
    with pytest.raises(ValueError, match="escapes image root"):
        compact_manifest(source, root, probe_ids=[1])
