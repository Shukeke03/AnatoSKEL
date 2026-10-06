"""Commands run only against explicit local files; no implicit data downloads."""
import argparse
import json
from pathlib import Path

import torch

from anatoskel.data import ScanDataset, generate_synthetic_dataset, label_to_class
from anatoskel.geometry import build_atlas, export_meshes, mesh_mask
from anatoskel.inference import evaluate_manifest, infer_manifest
from anatoskel.runtime import doctor
from anatoskel.training.pipeline import train_model, train_prior


def main():
    parser = argparse.ArgumentParser(prog="anatoskel")
    from anatoskel import __version__
    parser.add_argument("--version", action="version", version=f"AnatoSKEL {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("doctor", help="Test installed PyTorch operations")
    health.add_argument("--require-cuda", action="store_true")
    synthetic = commands.add_parser("make-synthetic", help="Create software-only NIfTI fixtures")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--size", type=int, default=32)
    inspect = commands.add_parser("inspect-data", help="Validate local NIfTI data and patient splits")
    inspect.add_argument("--manifest", required=True)
    atlas = commands.add_parser("build-atlas", help="Build a fixed frame gauge from training masks only")
    atlas.add_argument("--manifest", required=True)
    atlas.add_argument("--output", required=True)
    atlas.add_argument("--max-points", type=int, default=2048)
    atlas.add_argument("--cache-dir")
    atlas.add_argument("--label-ids", nargs="+", type=int)
    prior = commands.add_parser("train-prior", help="Fit a provisional canonical shape-prior warm-start")
    training = commands.add_parser("train", help="Train coarse detector and local reconstruction")
    for command in (prior, training):
        command.add_argument("--config", default="configs/research.yaml")
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="auto")
        command.add_argument("--max-steps", type=int)
    prior.add_argument("--epochs", type=int, default=20)
    prior.add_argument("--atlas", dest="atlas_path")
    training.add_argument("--prior")
    training.add_argument("--resume")
    training.add_argument("--allow-uninitialized-prior", action="store_true")
    versioned = commands.add_parser("train-v1", help="Train v1 shape/CT/diffusion/partial stage")
    versioned.add_argument("--config", default="configs/v1_research.yaml")
    versioned.add_argument("--manifest", required=True)
    versioned.add_argument("--output", required=True)
    versioned.add_argument("--stage", required=True, choices=("shape", "ct", "diffusion", "partial"))
    versioned.add_argument("--device", default="auto")
    versioned.add_argument("--init", help="Initialize next stage from a v1 checkpoint")
    versioned.add_argument("--resume", help="Resume the same stage, optimizer and cursor")
    versioned.add_argument("--max-steps", type=int)
    inference = commands.add_parser("infer", help="CT -> predicted ROI -> observed OBJ surfaces")
    evaluation = commands.add_parser("evaluate", help="Evaluate predicted-ROI Dice, including misses")
    for command in (inference, evaluation):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="auto")
        command.add_argument("--split", choices=("train", "val", "test"),
                             default="test" if command is inference else "val")
        command.add_argument("--allow-synthetic", action="store_true")
    inference.add_argument("--score-threshold", type=float)
    reference = commands.add_parser("export-reference", help="Export annotated masks; this is NOT model inference")
    reference.add_argument("--manifest", required=True)
    reference.add_argument("--output", required=True)
    reference.add_argument("--split", choices=("train", "val", "test"), default="test")
    arguments = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    values = vars(arguments).copy()
    command = values.pop("command")
    if command == "doctor":
        result = doctor(**values)
    elif command == "make-synthetic":
        result = {"manifest": str(generate_synthetic_dataset(values["output"], values["size"])),
                  "purpose": "synthetic software verification only"}
    elif command == "inspect-data":
        dataset = ScanDataset(values["manifest"])
        result = [{**scan["metadata"], "shape_dhw": list(scan["volume"].shape[-3:]),
                   "label_ids": [] if scan["labels"] is None else torch.unique(scan["labels"]).tolist()}
                  for scan in dataset]
    elif command == "train-prior":
        result = train_prior(values.pop("config"), **values)
    elif command == "build-atlas":
        result = {"atlas": str(build_atlas(values["manifest"], values["output"], values["max_points"],
                                          cache_dir=values["cache_dir"], label_ids=values["label_ids"])),
                  "anatomical_axes_validated": False}
    elif command == "train":
        values["prior_path"] = values.pop("prior")
        result = train_model(values.pop("config"), **values)
    elif command == "train-v1":
        from anatoskel.v1.pipeline import train_v1
        result = train_v1(values.pop("config"), **values)
    elif command == "infer":
        result = infer_manifest(**values)
    elif command == "evaluate":
        result = evaluate_manifest(**values)
    else:
        result = []
        for scan in ScanDataset(values["manifest"], values["split"]):
            if scan["labels"] is None:
                raise ValueError("Reference export requires annotated masks")
            meshes = [mesh_mask(scan["labels"] == int(label), scan["affine"], scan["valid_mask"],
                      name=f"reference_vertebra_{int(label)}", label_id=int(label),
                      class_id=label_to_class(int(label)))
                      for label in torch.unique(scan["labels"]) if int(label) > 0]
            paths = export_meshes(meshes, Path(values["output"]) / scan["metadata"]["scan_id"],
                                  metadata={**scan["metadata"], "source": "ground_truth_reference",
                                            "model_prediction": False})
            result.append({k: str(v) for k, v in paths.items()})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
