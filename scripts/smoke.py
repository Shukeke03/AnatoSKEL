"""Run a small software verification, without downloading patient data."""
from pathlib import Path
import json

from anatoskel.data import generate_synthetic_dataset
from anatoskel.inference import evaluate_manifest, infer_manifest
from anatoskel.runtime import doctor
from anatoskel.training.pipeline import train_model, train_prior
import subprocess
import sys
import torch


def run():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "smoke"
    config = root / "configs" / "smoke.yaml"
    torch.set_num_threads(2)
    health = doctor()
    manifest = generate_synthetic_dataset(out / "data")
    prior = train_prior(config, manifest, out / "prior", device="cpu", max_steps=2)
    model = train_model(config, manifest, out / "model", device="cpu",
                        prior_path=prior["checkpoint"], max_steps=1)
    predictions = infer_manifest(model["checkpoint"], manifest, out / "prediction",
                                 device="cpu", allow_synthetic=True)
    evaluation = evaluate_manifest(model["checkpoint"], manifest, out / "val.json",
                                    device="cpu", allow_synthetic=True)
    subprocess.run([sys.executable, "-m", "anatoskel.cli", "export-reference", "--manifest",
                    str(manifest), "--output", str(out / "reference")], cwd=root, check=True)
    report = {"purpose": "software verification only; not medical validation", "runtime": health,
              "prior": prior, "model": model, "prediction": predictions,
              "evaluation": evaluation}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
