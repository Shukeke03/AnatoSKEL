"""Exercise every v1 stage with ellipsoids; this is not a medical benchmark."""
from pathlib import Path
import json
import torch

from anatoskel.data import generate_synthetic_dataset
from anatoskel.v1.pipeline import train_v1
from anatoskel.v1.inference import infer_manifest


def main():
    torch.set_num_threads(4)
    root = Path("outputs/v1-smoke")
    manifest = generate_synthetic_dataset(root/"synthetic", size=32)
    previous = None
    results = []
    for stage in ("shape", "ct", "diffusion", "partial"):
        result = train_v1("configs/v1_smoke.yaml", manifest, root/stage,
                          stage=stage, device="cpu", init=previous, max_steps=2)
        results.append(result)
        previous = result["checkpoint"]
    result = infer_manifest(previous, manifest, root/"prediction", device="cpu",
                            allow_synthetic=True, score_threshold=0.)
    print(json.dumps({"stages": results, "prediction": result,
                      "purpose": "software workflow only; predictions may be empty or incorrect"}, indent=2))


if __name__ == "__main__":
    main()
