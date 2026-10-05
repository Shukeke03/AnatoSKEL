"""NIfTI voxel xyz is kept distinct from tensor DHW throughout this module.

The affine always maps NIfTI xyz voxel centres to patient RAS millimetres.
PCA frames are provisional software labels, not validated anatomical axes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import Dataset


VERSE_IDS = tuple(range(1, 25)) + (25, 28)
VERSE_CLASSES = {label: index for index, label in enumerate(VERSE_IDS)}


def label_to_class(label_id: int) -> int:
    """Map a foreground VerSe ID to contiguous class 0..25 (0 is background)."""
    try:
        return VERSE_CLASSES[int(label_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported foreground VerSe label {label_id}") from exc


@dataclass(frozen=True)
class ManifestRecord:
    patient_id: str
    scan_id: str
    split: str
    ct: Path
    mask: Path | None = None
    valid_mask: Path | None = None
    ct_valid_mask: Path | None = None


def _record(data: dict[str, Any], parent: Path) -> ManifestRecord:
    missing = {"patient_id", "scan_id", "split", "ct"} - set(data)
    if missing:
        raise ValueError(f"Manifest record misses {sorted(missing)}")
    for key in ("patient_id", "scan_id", "ct"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"Manifest {key} must be a nonempty string")
    if data["split"] not in ("train", "val", "test"):
        raise ValueError("Manifest split must be train, val, or test")

    def path(key):
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"Manifest {key} must be a path string")
        result = Path(value).expanduser()
        return (parent / result).resolve() if not result.is_absolute() else result.resolve()

    return ManifestRecord(data["patient_id"], data["scan_id"], data["split"],
                          path("ct"), path("mask"), path("valid_mask"), path("ct_valid_mask"))


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read JSONL and reject scan duplicates, CT reuse, and patient split leakage."""
    path = Path(path).resolve()
    records, scans, ct_paths, patients = [], set(), set(), {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
                if not isinstance(values, dict):
                    raise ValueError("Record must be an object")
                record = _record(values, path.parent)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if record.scan_id in scans or record.ct in ct_paths:
                raise ValueError(f"Duplicate scan or CT path: {record.scan_id}")
            if record.patient_id in patients and patients[record.patient_id] != record.split:
                raise ValueError(f"Patient leakage across splits: {record.patient_id}")
            scans.add(record.scan_id)
            ct_paths.add(record.ct)
            patients[record.patient_id] = record.split
            records.append(record)
    if not records:
        raise ValueError("Manifest has no records")
    return records


def validate_affine(affine, *, orthogonal: bool = False) -> np.ndarray:
    """Validate a voxel-centre affine; reflections are allowed, shear is optional."""
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], (0, 0, 0, 1), atol=1e-8):
        raise ValueError("Affine must have homogeneous last row [0,0,0,1]")
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 1e-8) or abs(np.linalg.det(basis)) <= 1e-10:
        raise ValueError("Affine basis must be invertible with positive voxel lengths")
    if orthogonal:
        directions = basis / spacing[None]
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
            raise ValueError("EDT requires orthogonal voxel axes; resample shear first")
    return affine


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _spacing(value: float | Sequence[float]) -> np.ndarray:
    result = np.repeat(float(value), 3) if np.isscalar(value) else np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("Spacing must contain three positive xyz millimetre values")
    return result


def _load_nifti(path: Path):
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI at {path}, got {image.shape}")
    units = image.header.get_xyzt_units()[0]
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if units not in factors:
        raise ValueError(f"Spatial units must be explicit mm/meter/micron at {path}; got {units}")
    affine = np.asarray(image.affine, dtype=np.float64).copy()
    affine[:3] *= factors[units]
    return image, validate_affine(affine)


def _truncated(mask_xyz: np.ndarray, valid_xyz: np.ndarray) -> bool:
    observed_interior = ndimage.binary_erosion(valid_xyz, border_value=0)
    return bool(np.any(mask_xyz & ~observed_interior))


def load_scan(record: ManifestRecord | dict[str, Any]) -> dict[str, Any]:
    """Load a scan. Dict paths are relative to the current directory.

    Returned affine maps xyz voxel centres, despite volume/labels being DHW.
    valid_mask is annotation validity; ct_valid_mask is actual CT coverage.
    """
    record = record if isinstance(record, ManifestRecord) else _record(record, Path.cwd())
    image, affine = _load_nifti(record.ct)
    ct = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(ct).all():
        raise ValueError("CT contains nonfinite intensities")
    labels = None
    if record.mask is not None:
        label_image, label_affine = _load_nifti(record.mask)
        if label_image.shape != image.shape or not np.allclose(label_affine, affine, atol=1e-5):
            raise ValueError("CT and mask must share voxel shape and physical affine")
        raw = np.asarray(label_image.dataobj)
        if not np.isfinite(raw).all() or not np.allclose(raw, np.rint(raw), atol=1e-6):
            raise ValueError("Mask must contain finite integer labels")
        labels = np.rint(raw).astype(np.int64)
        unknown = set(np.unique(labels)) - {0, *VERSE_IDS}
        if unknown:
            raise ValueError(f"Unsupported mask labels: {sorted(unknown)}")
    def load_coverage(path, kind):
        valid_image, valid_affine = _load_nifti(path)
        if valid_image.shape != image.shape or not np.allclose(valid_affine, affine, atol=1e-5):
            raise ValueError(f"CT and {kind} must share voxel shape and affine")
        raw_valid = np.asarray(valid_image.dataobj)
        if not np.isin(raw_valid, (0, 1)).all():
            raise ValueError(f"{kind} must be binary")
        return raw_valid.astype(bool)

    # CT coverage and annotation validity are different domains. Inference
    # must never use GT annotation holes to restrict observable reconstruction.
    ct_valid = np.ones(image.shape, dtype=bool) if record.ct_valid_mask is None else load_coverage(record.ct_valid_mask, "ct_valid_mask")
    valid = ct_valid.copy() if record.valid_mask is None else load_coverage(record.valid_mask, "valid_mask") & ct_valid
    if not ct_valid.any():
        raise ValueError("Scan has no observed voxels")
    if labels is not None and np.any((labels != 0) & ~valid):
        raise ValueError("Foreground labels may not lie outside valid_mask")
    truncation = {} if labels is None else {
        str(label): _truncated(labels == label, valid)
        for label in np.unique(labels) if label != 0
    }
    ct = 2 * (np.clip(ct, -1000, 2000) + 1000) / 3000 - 1
    metadata = {"patient_id": record.patient_id, "scan_id": record.scan_id,
                "split": record.split, "ct_path": str(record.ct), "units": "mm",
                "coordinate_system": "RAS", "original_shape_xyz": list(image.shape),
                "original_affine": affine.tolist(), "original_truncated": truncation,
                "ct_voxel_domain": {"shape_xyz": list(image.shape), "affine": affine.tolist()}}
    return {"volume": torch.from_numpy(ct.transpose(2, 1, 0).copy()).unsqueeze(0),
            "labels": None if labels is None else torch.from_numpy(labels.transpose(2, 1, 0).copy()),
            "valid_mask": torch.from_numpy(valid.transpose(2, 1, 0).copy()),
            "ct_valid_mask": torch.from_numpy(ct_valid.transpose(2, 1, 0).copy()),
            "artificial_padding_mask": torch.zeros(tuple(reversed(image.shape)), dtype=torch.bool),
            "affine": torch.from_numpy(affine), "metadata": metadata}


class ScanDataset(Dataset):
    def __init__(self, manifest: str | Path | Sequence[ManifestRecord], split: str | None = None):
        self.records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
        if split is not None and split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if split is not None:
            self.records = [record for record in self.records if record.split == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return load_scan(self.records[index])


def coarse_resample(scan: dict, spacing_mm: float | Sequence[float], order: int = 1) -> dict:
    """Resample to an orthogonal grid while retaining the observed physical extent.

    Even sheared input is explicitly regridded; labels and coverage use nearest
    neighbours. The orthogonal polar basis preserves input handedness.
    """
    affine = validate_affine(scan["affine"])
    spacing = _spacing(spacing_mm)
    old_shape = np.asarray(scan["volume"].shape[-3:][::-1])
    u, _, vh = np.linalg.svd(affine[:3, :3])
    rotation = u @ vh
    corners = np.array(np.meshgrid(*[(0, n - 1) for n in old_shape], indexing="ij")).reshape(3, -1).T
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    aligned = world @ rotation
    low, high = aligned.min(axis=0), aligned.max(axis=0)
    shape = np.ceil((high - low) / spacing - 1e-7).astype(int) + 1
    new_affine = np.eye(4)
    new_affine[:3, :3] = rotation * spacing[None]
    new_affine[:3, 3] = rotation @ low
    mapping = np.linalg.inv(affine) @ new_affine

    def transform(array, interpolation, fill):
        xyz = _numpy(array).transpose(2, 1, 0)
        result = ndimage.affine_transform(xyz, mapping[:3, :3], mapping[:3, 3],
                                         output_shape=tuple(shape), order=interpolation,
                                         mode="constant", cval=fill, prefilter=False)
        return torch.from_numpy(result.transpose(2, 1, 0).copy())

    valid = transform(scan["valid_mask"].to(torch.uint8), 0, 0).bool()
    ct_valid = transform(scan.get("ct_valid_mask", torch.ones_like(scan["valid_mask"])).to(torch.uint8), 0, 0).bool()
    source_extent = transform(torch.ones_like(scan["valid_mask"], dtype=torch.uint8), 0, 0).bool()
    prior_artificial = transform(scan.get("artificial_padding_mask", torch.zeros_like(scan["valid_mask"])).to(torch.uint8), 0, 1).bool()
    metadata = dict(scan["metadata"], spacing_mm=spacing.tolist(), resampled=True)
    return {"volume": transform(scan["volume"][0], order, -1).float().unsqueeze(0),
            "labels": None if scan["labels"] is None else transform(scan["labels"], 0, 0).long(),
            "valid_mask": valid & ct_valid, "ct_valid_mask": ct_valid,
            "artificial_padding_mask": prior_artificial | ~source_extent,
            "affine": torch.from_numpy(new_affine), "metadata": metadata}


def crop_at_center(scan: dict, center_world, size: int | Sequence[int] = 128,
                   spacing_mm: float | Sequence[float] | None = 1.0) -> dict:
    """Sample an exact centred physical ROI, including predicted-centre inputs.

    Size is DHW. Sampling occurs directly from the source, avoiding a full
    high-resolution resampling allocation. Original missing coverage and
    artificial padding remain distinct masks. Annotation validity and actual
    CT coverage are propagated independently.
    """
    source = scan
    size_dhw = np.repeat(int(size), 3) if np.isscalar(size) else np.asarray(size, dtype=int)
    if size_dhw.shape != (3,) or np.any(size_dhw <= 0):
        raise ValueError("Crop size must contain three positive DHW dimensions")
    center = _numpy(center_world).astype(float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Crop centre must be a finite RAS-mm xyz point")
    old_affine = validate_affine(source["affine"])
    affine = np.eye(4)
    if spacing_mm is None:
        affine[:3, :3] = old_affine[:3, :3]
    else:
        u, _, vh = np.linalg.svd(old_affine[:3, :3])
        affine[:3, :3] = (u @ vh) * _spacing(spacing_mm)[None]
    affine[:3, 3] = center - affine[:3, :3] @ ((size_dhw[::-1] - 1) / 2)
    mapping = np.linalg.inv(old_affine) @ affine

    def cut(value, interpolation, fill):
        result = ndimage.affine_transform(_numpy(value).transpose(2, 1, 0),
                                         mapping[:3, :3], mapping[:3, 3],
                                         output_shape=tuple(size_dhw[::-1]), order=interpolation,
                                         mode="constant", cval=fill, prefilter=False)
        return torch.from_numpy(result.transpose(2, 1, 0).copy())

    source_extent = cut(torch.ones_like(source["valid_mask"], dtype=torch.uint8), 0, 0).bool()
    artificial = cut(source.get("artificial_padding_mask", torch.zeros_like(source["valid_mask"])).to(torch.uint8), 0, 1).bool() | ~source_extent
    metadata = dict(source["metadata"], crop_center_world=center.tolist(),
                    crop_shape_dhw=size_dhw.tolist(), crop_affine=affine.tolist())
    ct_valid = cut(source.get("ct_valid_mask", torch.ones_like(source["valid_mask"])).to(torch.uint8), 0, 0).bool()
    return {"volume": cut(source["volume"][0], 1, -1).float().unsqueeze(0),
            "labels": None if source["labels"] is None else cut(source["labels"], 0, 0).long(),
            "valid_mask": cut(source["valid_mask"].to(torch.uint8), 0, 0).bool() & ct_valid,
            "ct_valid_mask": ct_valid,
            "artificial_padding_mask": artificial, "affine": torch.from_numpy(affine), "metadata": metadata}


def crop_bone(scan: dict, label_id: int, size: int | Sequence[int] = 128,
              spacing_mm: float | Sequence[float] | None = None, margin_mm: float = 8) -> dict:
    """Centre a physical crop on a GT visible centroid; inference uses crop_at_center."""
    label_to_class(label_id)
    if scan["labels"] is None:
        raise ValueError("crop_bone needs a mask; inference must use predicted ROIs")
    if margin_mm < 0:
        raise ValueError("Crop margin must be nonnegative")
    locations = torch.nonzero(scan["labels"] == label_id).cpu().numpy()
    if not len(locations):
        raise ValueError(f"Label {label_id} is absent")
    old_affine = validate_affine(scan["affine"])
    world = locations[:, ::-1] @ old_affine[:3, :3].T + old_affine[:3, 3]
    cropped = crop_at_center(scan, world.mean(axis=0), size=size, spacing_mm=spacing_mm)
    affine = validate_affine(cropped["affine"])
    inverse = np.linalg.inv(affine)
    new_voxels = world @ inverse[:3, :3].T + inverse[:3, 3]
    shape_xyz = np.asarray(cropped["valid_mask"].shape[::-1])
    lost = np.any((new_voxels < 0) | (new_voxels > shape_xyz - 1), axis=1)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    enough_margin = np.all(new_voxels.min(axis=0) * spacing >= margin_mm) and np.all(
        (shape_xyz - 1 - new_voxels.max(axis=0)) * spacing >= margin_mm)
    cropped["metadata"].update(crop_label_id=label_id, crop_artificial_boundary=bool(lost.any()),
                               crop_lost_bone_voxels=int(lost.sum()), crop_margin_satisfied=bool(enough_margin))
    return cropped


def mask_tsdf(mask, affine, distance_scale_mm: float = 8) -> torch.Tensor:
    """Negative-inside metric EDT, clipped and divided by the mm truncation band."""
    affine = validate_affine(affine, orthogonal=True)
    if not np.isfinite(distance_scale_mm) or distance_scale_mm <= 0:
        raise ValueError("Distance scale must be positive millimetres")
    mask = _numpy(mask).astype(bool)
    if mask.ndim != 3 or not mask.any() or mask.all():
        raise ValueError("TSDF needs a 3D mask with foreground and background")
    spacing_dhw = np.linalg.norm(affine[:3, :3], axis=0)[::-1]
    distance = ndimage.distance_transform_edt(~mask, sampling=spacing_dhw) - ndimage.distance_transform_edt(mask, sampling=spacing_dhw)
    return torch.from_numpy(np.clip(distance / distance_scale_mm, -1, 1).astype(np.float32))


def canonical_from_mask(mask, affine, template_points=None, min_voxels: int = 32,
                        partial: bool = False) -> dict[str, Any]:
    """Produce a deterministic provisional PCA frame, never an anatomical pose GT.

    For atlas fitting, use geometry.kabsch with actual matched point pairs. A
    point-cloud template alone has no correspondences and is deliberately not
    advertised as a completed registration procedure.
    """
    if template_points is not None:
        raise ValueError("Template correspondence fitting must use geometry.kabsch explicitly")
    affine = validate_affine(affine)
    voxels_dhw = np.argwhere(_numpy(mask).astype(bool))
    if len(voxels_dhw) < min_voxels:
        raise ValueError("Too few bone voxels for a stable provisional frame")
    world = voxels_dhw[:, ::-1] @ affine[:3, :3].T + affine[:3, 3]
    translation = world.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((world - translation).T))
    rotation = vectors[:, np.argsort(values)[::-1]]
    for column in (0, 1):
        axis = rotation[:, column]
        if axis[np.argmax(np.abs(axis))] < 0:
            rotation[:, column] *= -1
    rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
    eigenvalues = np.sort(values)[::-1]
    separation = float(min((eigenvalues[0] - eigenvalues[1]) / max(eigenvalues[0], 1e-8),
                           (eigenvalues[1] - eigenvalues[2]) / max(eigenvalues[1], 1e-8)))
    return {"rotation": torch.from_numpy(rotation), "translation": torch.from_numpy(translation),
            "pose_valid": False, "partial": bool(partial), "provisional": True,
            "method": "pca_initialization_only", "atlas_registration_done": False,
            "axis_separation": separation, "geometry_stable": separation > 0.05 and not partial}


def sample_supervision(scan: dict, label_id: int, n_points: int = 16384,
                       distance_scale_mm: float = 8, seed: int | None = None) -> dict:
    """Sample voxel-centre world points with surface/interior/exterior coverage.

    An entire truncation band next to unobserved/cropped data is censored to
    prevent EDT's artificial closing boundary from acting as a true surface.
    """
    class_id = label_to_class(label_id)
    if n_points < 1 or scan["labels"] is None:
        raise ValueError("Point sampling needs positive n_points and labels")
    mask = scan["labels"] == label_id
    target = mask_tsdf(mask, scan["affine"], distance_scale_mm)
    affine = validate_affine(scan["affine"], orthogonal=True)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)[::-1]
    valid = _numpy(scan["valid_mask"]).astype(bool)
    coverage_distance = ndimage.distance_transform_edt(np.pad(valid, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    reliable = valid & (coverage_distance > distance_scale_mm)
    target_numpy = _numpy(target)
    mask_numpy = _numpy(mask)
    boundary = reliable & (np.abs(target_numpy) < 0.5)
    populations = [np.argwhere(boundary), np.argwhere(reliable & mask_numpy),
                   np.argwhere(reliable & ~mask_numpy)]
    fallback = np.argwhere(reliable)
    if not len(fallback):
        raise ValueError("ROI has no uncensored supervision points; enlarge physical crop")
    rng = np.random.default_rng(seed)
    counts = [n_points // 2, n_points // 4, n_points - n_points // 2 - n_points // 4]
    indices = np.concatenate([population[rng.integers(len(population), size=count)]
                              if len(population) else fallback[rng.integers(len(fallback), size=count)]
                              for population, count in zip(populations, counts)])
    rng.shuffle(indices)
    points = indices[:, ::-1] @ affine[:3, :3].T + affine[:3, 3]
    values = target_numpy[tuple(indices.T)]
    return {"points_world": torch.from_numpy(points.astype(np.float32)),
            "target": torch.from_numpy(values.copy()), "valid": torch.ones(n_points, dtype=torch.bool),
            "class_id": class_id, "label_id": label_id,
            "voxel_dhw": torch.from_numpy(indices), "distance_scale_mm": distance_scale_mm}


def generate_synthetic_dataset(out_dir: str | Path, size: int = 32) -> Path:
    """Generate three independent ellipsoid fixtures solely for software checks."""
    if size < 16:
        raise ValueError("Synthetic grid size must be at least 16")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), axis=-1)
    affine = np.diag((1.1, 1.2, 1.3, 1.0))
    affine[:3, 3] = -np.array((1.1, 1.2, 1.3)) * (size - 1) / 2
    records = []
    for index, split in enumerate(("train", "val", "test")):
        labels = np.zeros((size,) * 3, dtype=np.uint16)
        for label_id, z in ((20, 0.32), (21, 0.68)):
            centre = np.array((0.5 + index * 0.008, 0.5, z)) * (size - 1)
            radii = np.array((0.15, 0.12, 0.115)) * size
            inside = np.sum(((grid - centre) / radii) ** 2, axis=-1) <= 1
            labels[inside] = label_id
        rng = np.random.default_rng(index)
        ct = (-800 + rng.normal(0, 8, labels.shape)).astype(np.float32)
        ct[labels != 0] = 600 + rng.normal(0, 10, np.count_nonzero(labels))
        patient_id, scan_id = f"synthetic_patient_{index}", f"synthetic_{split}"
        for suffix, array in (("ct", ct), ("mask", labels)):
            image = nib.Nifti1Image(array, affine)
            image.header.set_xyzt_units("mm")
            nib.save(image, out_dir / f"{scan_id}_{suffix}.nii.gz")
        records.append({"patient_id": patient_id, "scan_id": scan_id, "split": split,
                        "ct": f"{scan_id}_ct.nii.gz", "mask": f"{scan_id}_mask.nii.gz"})
    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    return manifest
