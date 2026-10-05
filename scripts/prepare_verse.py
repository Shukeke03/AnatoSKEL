#!/usr/bin/env python3
"""Import already-extracted VerSe-complete archives; no dataset downloads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anatoskel.data.verse import prepare_verse, VerseImportError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Parent of extracted dataset-verse* directories")
    parser.add_argument("--manifest", required=True, help="Output JSONL manifest")
    parser.add_argument("--report", help="Output provenance/dedup report JSON")
    parser.add_argument("--require-six-packages", action="store_true")
    parser.add_argument("--allow-missing-centroids", action="store_true")
    parser.add_argument("--no-annotation-domain", action="store_true",
                        help="Explicitly disable conservative terminal annotation censorship")
    parser.add_argument("--cross-split-policy", choices=("error", "holdout-priority"), default="error",
                        help="Default fails closed; explicit holdout-priority drops lower-split patient copies")
    parser.add_argument("--patient-map", help="Optional explicit JSON subject-to-patient identity mapping")
    args = parser.parse_args()
    try:
        report = prepare_verse(args.root, args.manifest, report=args.report,
                               require_six_packages=args.require_six_packages,
                               require_annotations=not args.allow_missing_centroids,
                               annotation_domain=not args.no_annotation_domain,
                               cross_split_policy=args.cross_split_policy, patient_map=args.patient_map,
                               progress=lambda message: print(message, flush=True))
    except VerseImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("status", "retained_scans", "retained_patients",
                                                   "split_counts", "manifest")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
