"""Low-memory importer for the official six VerSe-complete BIDS archives.

The subject-based release already removes the cross-year challenge overlap.
Patient identity is the BIDS ``sub-verseNNN`` or ``sub-glNNN`` entity, not the original scan ID
in ``split-verseNNN``. No random splitting or year-based patient offset occurs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import nibabel as nib
import numpy as np

from .scans import VERSE_IDS, nifti_affine_mm, validate_affine


OFFICIAL_SOURCES = {
    "release": "https://github.com/anjany/verse#download",
    "split_layout": "https://github.com/anjany/verse#data",
    "overlap_excluded": "https://github.com/anjany/verse/issues/1#issuecomment-870724814",
    "subject_release_merge": "https://github.com/anjany/verse/issues/6#issuecomment-982504561",
    "paper": "https://arxiv.org/abs/2001.09193",
}
PACKAGE_PATTERN = re.compile(r"dataset-verse(19|20)(training|validation|test)$", re.I)
SUBJECT_PATTERN = re.compile(r"sub-(verse|gl)(\d+)(?=_|$)", re.I)
SPLIT_MAP = {"training": "train", "validation": "val", "test": "test"}


class VerseImportError(ValueError):
    """An import failed closed; its report explains the conflict or validation error."""


def _ignored(path: Path) -> bool:
    return "__MACOSX" in path.parts or any(part.startswith("._") for part in path.parts)


def _subject(prefix: str) -> str:
    match = SUBJECT_PATTERN.match(prefix)
    if not match:
        raise VerseImportError(f"Not a subject-based BIDS VerSe scan: {prefix}")
    return f"sub-{match[1].lower()}{int(match[2]):03d}"


def discover_verse(root: str | Path, *, require_annotations: bool = True) -> list[dict[str, Any]]:
    """Find rawdata CTs and exact-entity derivative pairs in published package roots."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise VerseImportError(f"Extracted data directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not _ignored(path))
    index = defaultdict(list)
    for path in files:
        for parent in (path.parent, *path.parents):
            package_match = PACKAGE_PATTERN.fullmatch(parent.name)
            if package_match:
                index[parent].append(path)
                break
    if not index:
        raise VerseImportError("Expected extracted dataset-verse19/20training/validation/test directories")
    records = []
    for package, package_files in sorted(index.items()):
        match = PACKAGE_PATTERN.fullmatch(package.name)
        year, split = 2000 + int(match[1]), SPLIT_MAP[match[2].lower()]
        derivatives = [path for path in package_files if "derivatives" in path.relative_to(package).parts]
        for ct in package_files:
            if "rawdata" not in ct.relative_to(package).parts or not ct.name.endswith(("_ct.nii", "_ct.nii.gz")):
                continue
            stem = ct.name[:-7] if ct.name.endswith(".nii.gz") else ct.name[:-4]
            prefix = stem[:-3]
            subject = _subject(prefix)
            # The directory groups acquisitions such as split-verse201/253
            # under the canonical patient sub-verse401.
            parent_subjects = [part for part in ct.relative_to(package).parts[:-1] if SUBJECT_PATTERN.fullmatch(part)]
            if not parent_subjects or _subject(parent_subjects[-1]) != subject:
                raise VerseImportError(f"CT patient directory and filename disagree: {ct}")
            masks = [path for path in derivatives if path.name in
                     (f"{prefix}_seg-vert_msk.nii", f"{prefix}_seg-vert_msk.nii.gz")]
            if len(masks) != 1:
                raise VerseImportError(f"Expected one exact-entity vertebra mask for {ct}; found {len(masks)}")
            annotations = [path for path in derivatives if path.name.startswith(prefix + "_seg-")
                           and path.name.endswith("_ctd.json")]
            if require_annotations and not annotations:
                raise VerseImportError(f"Missing centroid annotation for {ct}")
            for annotation in annotations:
                _validate_centroids(annotation)
            records.append({"patient_id": subject, "scan_id": SUBJECT_PATTERN.sub(subject, prefix, count=1),
                            "split": split, "ct": ct, "mask": masks[0], "annotations": sorted(annotations),
                            "source_package": package.name, "source_year": year,
                            "original_scan_entity": re.search(r"(?:^|_)split-(verse\d+)", prefix).group(1)
                            if re.search(r"(?:^|_)split-(verse\d+)", prefix) else None})
    if not records:
        raise VerseImportError("No subject-based rawdata CT/mask pairs found")
    return sorted(records, key=lambda record: (record["source_package"], record["scan_id"], str(record["ct"])))


def _validate_centroids(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list):
        raise VerseImportError(f"Centroid annotation must be a list: {path}")
    for entry in values:
        if not isinstance(entry, dict):
            raise VerseImportError(f"Centroid entries must be objects: {path}")
        if "label" not in entry:
            if "direction" not in entry:
                raise VerseImportError(f"Unknown centroid metadata entry: {path}")
            continue
        if entry["label"] not in VERSE_IDS or not all(key in entry for key in ("X", "Y", "Z")):
            raise VerseImportError(f"Invalid centroid label/coordinates: {path}")
        try:
            coordinates = np.array([entry[key] for key in ("X", "Y", "Z")], dtype=float)
        except (TypeError, ValueError) as exc:
            raise VerseImportError(f"Invalid centroid coordinates: {path}") from exc
        if not np.isfinite(coordinates).all():
            raise VerseImportError(f"Nonfinite centroid coordinates: {path}")


def _open_image(path, *, reference=None):
    image = nib.load(str(path), keep_file_open=True)
    if len(image.shape) != 3:
        raise VerseImportError(f"Expected a 3D NIfTI: {path}")
    try:
        return image, nifti_affine_mm(image, reference=reference)
    except ValueError as exc:
        raise VerseImportError(f"Invalid physical NIfTI geometry: {path}: {exc}") from exc


def _chunks(image, chunk_slices=4):
    # NIfTI xyz/F-order means consecutive z planes have contiguous payload.
    # ArrayProxy never materializes the full CT or an int64 full-size mask.
    plane_bytes = image.shape[0] * image.shape[1] * 8
    chunk_slices = max(1, min(chunk_slices, (16 * 1024 * 1024) // max(1, plane_bytes)))
    for start in range(0, image.shape[2], chunk_slices):
        stop = min(image.shape[2], start + chunk_slices)
        yield start, stop, np.asanyarray(image.dataobj[:, :, start:stop])


def fingerprint_nifti(path, *, labels=False, chunk_slices=4, reference=None) -> dict[str, Any]:
    """Content SHA256 of scaled voxels plus voxel geometry, independent of gzip/header trivia.

    CT values canonicalize to little-endian float32; integer labels to uint16.
    This detects exact voxel/geometry duplicates, not resampled near-duplicates.
    """
    image, affine = _open_image(path, reference=reference)
    geometry = {"shape_xyz": list(image.shape), "affine_ras_mm": np.round(affine, 6).tolist()}
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode())
    seen_labels, foreground_count = set(), 0
    lower, upper = float("inf"), -float("inf")
    for start, stop, values in _chunks(image, chunk_slices):
        if not np.isfinite(values).all():
            raise VerseImportError(f"Nonfinite image values: {path}")
        if labels:
            if not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
                raise VerseImportError(f"Noninteger mask labels: {path}")
            unique = set(np.unique(values).astype(int).tolist())
            if unique - {0, *VERSE_IDS}:
                raise VerseImportError(f"Unsupported mask labels {sorted(unique - {0, *VERSE_IDS})}: {path}")
            seen_labels.update(unique)
            foreground = np.nonzero(values)
            count = len(foreground[0])
            foreground_count += count
            if count:
                world_z = (affine[2, 0] * foreground[0] + affine[2, 1] * foreground[1]
                           + affine[2, 2] * (foreground[2] + start) + affine[2, 3])
                lower, upper = min(lower, float(world_z.min())), max(upper, float(world_z.max()))
            payload = np.asarray(np.rint(values), dtype="<u2")
        else:
            payload = np.asarray(values, dtype="<f4")
        digest.update(payload.tobytes(order="F"))
    result = {"sha256": digest.hexdigest(), "shape_xyz": list(image.shape),
              "affine_ras_mm": affine.tolist(), "dtype_policy": "uint16" if labels else "float32",
              "fingerprint_scope": "scaled_xyz_voxels_and_mm_affine_rounded_1e-6",
              "source_spatial_units": image.header.get_xyzt_units()[0],
              "spatial_units_policy": "inherited_from_identical_ct_grid" if
              image.header.get_xyzt_units()[0] == "unknown" else "explicit_header"}
    if labels:
        if not foreground_count:
            raise VerseImportError(f"No annotated foreground in mask: {path}")
        result.update(labels=sorted(seen_labels - {0}), foreground_voxels=foreground_count,
                      annotated_superior_range_mm=[lower, upper])
    return result


def write_annotation_domain(shape_xyz, affine, superior_range_mm, path: str | Path) -> Path:
    """Stream a conservative annotation slab to uint8 .nii.gz using bounded memory.

    Only the superior-coordinate envelope of labeled bone is included. This
    avoids treating unannotated terminal vertebrae as background; it is a
    conservative supervision policy, not a ground-truth observation mask.
    """
    shape = tuple(int(value) for value in shape_xyz)
    affine = validate_affine(affine)
    low, high = map(float, superior_range_mm)
    if not np.isfinite([low, high]).all() or low >= high:
        raise VerseImportError("Annotation superior envelope must have positive extent")
    header = nib.Nifti1Header()
    header.set_data_shape(shape)
    header.set_data_dtype(np.uint8)
    header.set_xyzt_units("mm")
    header.set_sform(affine, code=1)
    header.set_qform(affine, code=1, strip_shears=True)
    header.set_slope_inter(1, 0)
    header["vox_offset"] = 352
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    xy = affine[2, 0] * np.arange(shape[0])[:, None] + affine[2, 1] * np.arange(shape[1])[None, :] + affine[2, 3]
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        stream.write(header.binaryblock)
        stream.write(b"\x00\x00\x00\x00")
        for z in range(shape[2]):
            superior = xy + affine[2, 2] * z
            valid = (superior >= low - 1e-5) & (superior <= high + 1e-5)
            stream.write(np.asarray(valid, dtype=np.uint8).tobytes(order="F"))
    temporary.replace(path)
    return path


class _Patients:
    def __init__(self):
        self.parents = {}

    def root(self, patient):
        self.parents.setdefault(patient, patient)
        if self.parents[patient] != patient:
            self.parents[patient] = self.root(self.parents[patient])
        return self.parents[patient]

    def merge(self, one, two):
        left, right = self.root(one), self.root(two)
        if left != right:
            low, high = sorted((left, right))
            self.parents[high] = low


def prepare_verse(root: str | Path, manifest: str | Path, *, report: str | Path | None = None,
                  require_six_packages: bool = False, require_annotations: bool = True,
                  annotation_domain: bool = True, cross_split_policy: str = "error",
                  patient_map: str | Path | None = None, progress=None) -> dict[str, Any]:
    """Validate and import extracted official subject-based archives, failing closed.

    Optional ``holdout-priority`` drops lower-priority copies/acquisitions of a
    patient (test > val > train); it never moves scans or creates a random split.
    That is an explicit custom subset policy, not an official benchmark split.
    """
    if cross_split_policy not in ("error", "holdout-priority"):
        raise ValueError("cross_split_policy must be error or holdout-priority")
    manifest = Path(manifest).resolve()
    report_path = Path(report).resolve() if report else manifest.with_suffix(".provenance.json")
    payload = {"status": "preparing", "sources": OFFICIAL_SOURCES,
               "release_format": "complete_subject_based_BIDS", "split_policy": cross_split_policy,
               "patient_identity": "BIDS sub entity shared across years; split entity identifies scan",
               "annotation_domain_policy": "labeled_bone_superior_envelope" if annotation_domain else "all_ct_unchecked_terminals",
               "discovered_scans": 0, "duplicates": [], "conflicts": [], "dropped": []}
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    try:
        records = discover_verse(root, require_annotations=require_annotations)
        payload["discovered_scans"] = len(records)
        payload["source_package_counts"] = dict(Counter(record["source_package"] for record in records))
        if require_six_packages and set(payload["source_package_counts"]) != {
            f"dataset-verse{year}{split}" for year in (19, 20) for split in SPLIT_MAP}:
            raise VerseImportError("All six extracted VerSe-complete packages are required")
        mapping = {} if patient_map is None else json.loads(Path(patient_map).read_text())
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in mapping.items()):
            raise VerseImportError("Explicit patient map must be a subject-to-patient string mapping")
        patients, by_content = _Patients(), defaultdict(list)
        for number, record in enumerate(records, 1):
            record["original_patient_id"] = record["patient_id"]
            record["patient_id"] = mapping.get(record["patient_id"], record["patient_id"])
            patients.root(record["patient_id"])
            if progress:
                progress(f"[{number}/{len(records)}] stream-check {record['scan_id']} ({record['split']})")
            ct_info = fingerprint_nifti(record["ct"])
            ct_image, _ = _open_image(record["ct"])
            mask_info = fingerprint_nifti(record["mask"], labels=True, reference=ct_image)
            if ct_info["shape_xyz"] != mask_info["shape_xyz"] or not np.allclose(
                ct_info["affine_ras_mm"], mask_info["affine_ras_mm"], atol=1e-5, rtol=0):
                raise VerseImportError(f"CT/mask physical grids differ: {record['scan_id']}")
            record.update(ct_info=ct_info, mask_info=mask_info)
            by_content[ct_info["sha256"]].append(record)
        for copies in by_content.values():
            for copy in copies[1:]:
                patients.merge(copies[0]["patient_id"], copy["patient_id"])
        grouped = defaultdict(list)
        for record in records:
            record["patient_id"] = patients.root(record["patient_id"])
            grouped[record["patient_id"]].append(record)
        for patient, scans in grouped.items():
            splits = sorted({record["split"] for record in scans})
            if len(splits) > 1:
                payload["conflicts"].append({"patient_id": patient, "splits": splits,
                                             "scans": [record["scan_id"] for record in scans]})
        if payload["conflicts"] and cross_split_policy == "error":
            raise VerseImportError("Patient/content overlap across official splits; review provenance report")
        rank = {"train": 0, "val": 1, "test": 2}
        selected = []
        for patient, scans in grouped.items():
            retained_split = max((record["split"] for record in scans), key=rank.get)
            for record in scans:
                if record["split"] == retained_split:
                    selected.append(record)
                else:
                    payload["dropped"].append({"scan_id": record["scan_id"], "patient_id": patient,
                                               "split": record["split"], "reason": "explicit_holdout_priority"})
        kept, seen, seen_ids = [], {}, {}
        for record in sorted(selected, key=lambda item: (item["source_package"], item["scan_id"], str(item["ct"]))):
            key = record["ct_info"]["sha256"]
            if key in seen:
                previous = seen[key]
                if previous["mask_info"]["sha256"] != record["mask_info"]["sha256"]:
                    payload["conflicts"].append({"reason": "duplicate_ct_different_masks",
                                                 "scans": [previous["scan_id"], record["scan_id"]]})
                    raise VerseImportError("Identical CT has conflicting mask versions; no version chosen automatically")
                payload["duplicates"].append({"kept": str(previous["ct"]), "discarded": str(record["ct"]),
                                               "ct_sha256": key, "split": record["split"],
                                               "patient_alias": record["original_patient_id"]})
                continue
            if record["scan_id"] in seen_ids:
                raise VerseImportError(f"Same BIDS scan ID has different CT content: {record['scan_id']}")
            seen[key], seen_ids[record["scan_id"]] = record, key
            kept.append(record)
        output = []
        for record in sorted(kept, key=lambda item: (item["split"], item["patient_id"], item["scan_id"])):
            def relative(path):
                return os.path.relpath(Path(path).resolve(), manifest.parent)
            row = {key: record[key] for key in ("patient_id", "scan_id", "split", "source_package", "source_year")}
            row.update(ct=relative(record["ct"]), mask=relative(record["mask"]),
                       ct_sha256=record["ct_info"]["sha256"], mask_sha256=record["mask_info"]["sha256"],
                       mask_source_spatial_units=record["mask_info"]["source_spatial_units"],
                       mask_spatial_units_policy=record["mask_info"]["spatial_units_policy"],
                       original_patient_id=record["original_patient_id"],
                       original_scan_entity=record["original_scan_entity"],
                       centroid_annotations=[relative(path) for path in record["annotations"]])
            if annotation_domain:
                valid_path = manifest.parent / "annotation_valid" / f"{record['scan_id']}_valid.nii.gz"
                write_annotation_domain(record["mask_info"]["shape_xyz"], record["mask_info"]["affine_ras_mm"],
                                        record["mask_info"]["annotated_superior_range_mm"], valid_path)
                row["valid_mask"] = relative(valid_path)
            output.append(row)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_name(manifest.name + ".tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output))
        temporary.replace(manifest)
        payload.update(status="complete", retained_scans=len(output),
                       retained_patients=len({row["patient_id"] for row in output}),
                       split_counts=dict(Counter(row["split"] for row in output)),
                       split_patient_counts={split: len({row["patient_id"] for row in output if row["split"] == split})
                                             for split in ("train", "val", "test")},
                       manifest=str(manifest), fingerprint_method="streamed_scaled_voxels_plus_xyz_mm_geometry",
                       patient_remapping=[{"scan_id": row["scan_id"], "patient_id": row["patient_id"],
                                           "original_patient_id": row["original_patient_id"],
                                           "original_scan_entity": row["original_scan_entity"]} for row in output])
        save_report()
        return payload
    except (ValueError, OSError, nib.filebasedimages.ImageFileError) as exc:
        payload.update(status="failed", error=str(exc))
        save_report()
        raise VerseImportError(f"{exc}; report: {report_path}") from exc
