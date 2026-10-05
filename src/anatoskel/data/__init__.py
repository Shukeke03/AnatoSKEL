"""Physical-coordinate NIfTI data, explicit splits, and mask supervision."""
from .scans import (
    ManifestRecord, ScanDataset, canonical_from_mask, coarse_resample, crop_at_center, crop_bone,
    generate_synthetic_dataset, label_to_class, load_scan, mask_tsdf,
    read_manifest, sample_supervision, validate_affine,
)

__all__ = ["ManifestRecord", "ScanDataset", "canonical_from_mask", "coarse_resample",
           "crop_at_center", "crop_bone", "generate_synthetic_dataset", "label_to_class", "load_scan",
           "mask_tsdf", "read_manifest", "sample_supervision", "validate_affine"]
