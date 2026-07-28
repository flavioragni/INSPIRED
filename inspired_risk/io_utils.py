from __future__ import annotations

import json
from pathlib import Path

from .schemas import ClinicalFeatures, EyeInput, PatientInput


def load_patient_json(path: str | Path) -> PatientInput:
    path = Path(path)
    data = json.loads(path.read_text())
    patient = PatientInput(
        patient_id=data["patient_id"],
        eyes=[EyeInput(image_path=Path(e["image_path"]), eye_id=e.get("eye_id", f"eye_{i+1}")) for i, e in enumerate(data["eyes"])],
        clinical=ClinicalFeatures(**data["clinical"]),
    )
    patient.validate()
    return patient
