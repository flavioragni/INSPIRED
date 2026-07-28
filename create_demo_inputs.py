from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import numpy as np


def make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    example_dir = Path("examples")
    example_dir.mkdir(exist_ok=True)

    left = example_dir / "left_eye.png"
    right = example_dir / "right_eye.png"
    make_image(left, seed=1)
    make_image(right, seed=2)

    patient = {
        "patient_id": "demo_patient_001",
        "eyes": [
            {"eye_id": "left", "image_path": str(left.resolve())},
            {"eye_id": "right", "image_path": str(right.resolve())},
        ],
        "clinical": {
            "diabetes_type": "T2D",
            "diabetes_duration_years": 14,
            "hba1c": 7.8,
            "systolic_bp": 142,
            "sex": "male",
            "egfr": 82,
            "fenofibrate": False,
            "anemia": True,
        },
    }
    (example_dir / "patient_demo.json").write_text(json.dumps(patient, indent=2))
    print(f"Wrote example files under {example_dir.resolve()}")


if __name__ == "__main__":
    main()
