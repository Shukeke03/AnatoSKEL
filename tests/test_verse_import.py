"""Synthetic BIDS trees test importer safety; no real VerSe data is downloaded."""
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from anatoskel.data import load_scan, read_manifest
from anatoskel.data.verse import (VerseImportError, discover_verse, fingerprint_nifti,
                                  prepare_verse, write_annotation_domain)


def case(root, package, subject, suffix="", seed=1, label=20, affine=None):
    affine = np.diag([1.0, 1.2, 1.5, 1.0]) if affine is None else affine
    prefix = subject + suffix
    raw = root / package / "rawdata" / subject
    derivatives = root / package / "derivatives" / subject
    raw.mkdir(parents=True, exist_ok=True)
    derivatives.mkdir(parents=True, exist_ok=True)
    x, y, z = np.indices((16, 16, 16))
    ct = (x + 2 * y + 3 * z + seed).astype(np.int16)
    mask = np.zeros(ct.shape, np.uint8)
    mask[5:11, 5:11, 5:11] = label
    paths = (raw / f"{prefix}_ct.nii.gz", derivatives / f"{prefix}_seg-vert_msk.nii.gz")
    for path, array in zip(paths, (ct, mask)):
        image = nib.Nifti1Image(array, affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
    annotation = derivatives / f"{prefix}_seg-subreg_ctd.json"
    annotation.write_text(json.dumps([{"direction": ["R", "A", "S"]},
                                      {"label": label, "X": 8, "Y": 8, "Z": 8}]))
    return paths


def test_preserves_six_official_splits_and_patient_scan_entities(tmp_path):
    raw = tmp_path / "raw"
    counter = 0
    for year in (19, 20):
        for split in ("training", "validation", "test"):
            counter += 1
            case(raw, f"dataset-verse{year}{split}", f"sub-verse{counter:03d}", seed=counter)
    case(raw, "dataset-verse19training", "sub-verse401", "_split-verse201", seed=201)
    case(raw, "dataset-verse19training", "sub-verse401", "_split-verse253", seed=253)
    rubbish = raw / "__MACOSX" / "dataset-verse19training" / "rawdata" / "sub-verse000"
    rubbish.mkdir(parents=True)
    (rubbish / "sub-verse000_ct.nii.gz").write_text("not an image")
    (raw / "dataset-verse19training" / "rawdata" / "sub-verse001" / "._sub-verse001_ct.nii.gz").write_text("resource fork")
    manifest = tmp_path / "processed" / "manifest.jsonl"
    report = prepare_verse(raw, manifest, require_six_packages=True)
    records = read_manifest(manifest)
    assert len(records) == 8 and report["retained_patients"] == 7
    assert report["split_counts"] == {"train": 4, "val": 2, "test": 2}
    same_patient = [record for record in records if record.patient_id == "sub-verse401"]
    assert len(same_patient) == 2 and all(record.split == "train" for record in same_patient)
    assert {record.scan_id for record in same_patient} == {"sub-verse401_split-verse201", "sub-verse401_split-verse253"}
    assert all(record.valid_mask.exists() for record in records)
    scan = load_scan(same_patient[0])
    assert scan["ct_valid_mask"].all() and not scan["valid_mask"].all()
    assert scan["valid_mask"][scan["labels"] != 0].all()
    identities = {entry["original_scan_entity"] for entry in report["patient_remapping"] if entry["patient_id"] == "sub-verse401"}
    assert identities == {"verse201", "verse253"}


def test_gl_subjects_preserve_namespace_and_official_split(tmp_path):
    raw = tmp_path / "raw"
    case(raw, "dataset-verse19training", "sub-verse003", seed=1)
    case(raw, "dataset-verse20test", "sub-gl003", seed=2)
    case(raw, "dataset-verse20validation", "sub-GL004", seed=3)
    manifest = tmp_path / "manifest.jsonl"
    report = prepare_verse(raw, manifest)
    records = read_manifest(manifest)
    assert report["retained_patients"] == 3
    assert {record.patient_id: record.split for record in records} == {
        "sub-verse003": "train", "sub-gl003": "test", "sub-gl004": "val"}
    assert {record.scan_id for record in records} == {
        "sub-verse003", "sub-gl003", "sub-gl004"}


def test_missing_mask_units_import_and_load_with_paired_ct(tmp_path):
    raw = tmp_path / "raw"
    ct, mask = case(raw, "dataset-verse19training", "sub-verse004")
    explicit_info = fingerprint_nifti(mask, labels=True)
    image = nib.load(mask)
    image.header.set_xyzt_units("unknown")
    # Save to a different path before replacing the source read by ArrayProxy.
    replacement = mask.with_name("replacement.nii.gz")
    nib.save(image, replacement)
    replacement.replace(mask)
    inherited_info = fingerprint_nifti(mask, labels=True, reference=nib.load(ct))
    assert inherited_info["sha256"] == explicit_info["sha256"]
    assert inherited_info["spatial_units_policy"] == "inherited_from_identical_ct_grid"
    with pytest.raises(VerseImportError, match="Spatial units"):
        fingerprint_nifti(mask, labels=True)
    manifest = tmp_path / "manifest.jsonl"
    prepare_verse(raw, manifest)
    row = json.loads(manifest.read_text())
    assert row["mask_source_spatial_units"] == "unknown"
    assert row["mask_spatial_units_policy"] == "inherited_from_identical_ct_grid"
    scan = load_scan(read_manifest(manifest)[0])
    assert (scan["labels"] == 20).sum().item() == 216
    assert nib.load(mask).header.get_xyzt_units()[0] == "unknown"


def test_fingerprint_streams_without_full_proxy_array_and_normalizes_dtype(tmp_path, monkeypatch):
    ct, _ = case(tmp_path, "dataset-verse19training", "sub-verse004")
    image = nib.load(ct)
    values = np.asarray(image.dataobj, dtype=np.float32)
    twin = tmp_path / "different_header.nii.gz"
    copied = nib.Nifti1Image(values, image.affine)
    copied.header.set_xyzt_units("mm")
    copied.header["descrip"] = b"different header and floating storage"
    nib.save(copied, twin)

    def forbidden_full_array(*args, **kwargs):
        raise AssertionError("Importer must not materialize a full ArrayProxy")

    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__array__", forbidden_full_array)
    monkeypatch.setattr(nib.Nifti1Image, "get_fdata", forbidden_full_array)
    one = fingerprint_nifti(ct, chunk_slices=1)
    two = fingerprint_nifti(twin, chunk_slices=3)
    assert one["sha256"] == two["sha256"]


def test_content_alias_dedup_within_split_and_cross_split_fails_closed(tmp_path):
    raw = tmp_path / "raw"
    case(raw, "dataset-verse19training", "sub-verse004", seed=9)
    case(raw, "dataset-verse20training", "sub-verse504", seed=9)
    manifest = tmp_path / "manifest.jsonl"
    result = prepare_verse(raw, manifest)
    assert result["retained_scans"] == 1 and len(result["duplicates"]) == 1
    assert read_manifest(manifest)[0].patient_id == "sub-verse004"
    previous = manifest.read_text()
    case(raw, "dataset-verse20test", "sub-verse604", seed=9)
    with pytest.raises(VerseImportError, match="overlap across"):
        prepare_verse(raw, manifest)
    assert manifest.read_text() == previous  # Failure never replaces a valid manifest.
    failure = json.loads(manifest.with_suffix(".provenance.json").read_text())
    assert failure["status"] == "failed" and failure["conflicts"]
    explicit = prepare_verse(raw, manifest, cross_split_policy="holdout-priority")
    assert explicit["split_counts"] == {"test": 1} and len(explicit["dropped"]) == 2


def test_patient_cross_split_conflict_and_mask_version_conflict(tmp_path):
    raw = tmp_path / "raw"
    case(raw, "dataset-verse19training", "sub-verse004", seed=1)
    case(raw, "dataset-verse20validation", "sub-verse004", "_acq-other", seed=2)
    with pytest.raises(VerseImportError, match="overlap across"):
        prepare_verse(raw, tmp_path / "conflict.jsonl")
    other = tmp_path / "other"
    case(other, "dataset-verse19training", "sub-verse004", seed=1)
    case(other, "dataset-verse20training", "sub-verse504", seed=1, label=21)
    with pytest.raises(VerseImportError, match="conflicting mask"):
        prepare_verse(other, tmp_path / "mask_conflict.jsonl")


def test_mask_pair_labels_and_required_package_validation(tmp_path):
    raw = tmp_path / "raw"
    _, mask = case(raw, "dataset-verse19training", "sub-verse004")
    with pytest.raises(VerseImportError, match="All six"):
        prepare_verse(raw, tmp_path / "manifest.jsonl", require_six_packages=True)
    mask.unlink()
    with pytest.raises(VerseImportError, match="one exact-entity"):
        discover_verse(raw)
    bad = tmp_path / "bad"
    case(bad, "dataset-verse19training", "sub-verse005", label=26)
    with pytest.raises(VerseImportError, match="Invalid centroid"):
        prepare_verse(bad, tmp_path / "bad.jsonl")


def test_annotation_slab_uses_patient_superior_axis_not_array_depth(tmp_path):
    affine = np.array([[1, 0, 0, 0], [0, 0, 2, -10], [0, -3, 0, 30], [0, 0, 0, 1]], dtype=float)
    path = write_annotation_domain((10, 12, 14), affine, [9, 18], tmp_path / "valid.nii.gz")
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    np.testing.assert_allclose(image.affine, affine)
    assert values.dtype == np.uint8 and image.header.get_xyzt_units()[0] == "mm"
    assert values[:, 4:8, :].all()
    assert not values[:, :4, :].any() and not values[:, 8:, :].any()


def test_standalone_cli_imports_without_editing_main_cli(tmp_path):
    raw = tmp_path / "raw"
    case(raw, "dataset-verse19training", "sub-verse004")
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_verse.py"
    manifest = tmp_path / "manifest.jsonl"
    result = subprocess.run([sys.executable, str(script), "--root", str(raw),
                             "--manifest", str(manifest)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert manifest.exists() and "retained_scans" in result.stdout
