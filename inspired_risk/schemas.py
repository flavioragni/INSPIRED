from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

DiabetesType = Literal["T1D", "T2D"]
SexType = Literal["female", "male"]


@dataclass
class ClinicalFeatures:
    diabetes_type: Optional[DiabetesType] = None
    diabetes_duration_years: Optional[float] = None
    hba1c: Optional[float] = None
    systolic_bp: Optional[float] = None
    sex: Optional[SexType] = None
    egfr: Optional[float] = None
    fenofibrate: Optional[bool] = None
    anemia: Optional[bool] = None
    bmi: Optional[float] = None


@dataclass
class EyeInput:
    image_path: Path
    eye_id: str = "unknown_eye"


@dataclass
class PatientInput:
    patient_id: str
    eyes: list[EyeInput]
    clinical: ClinicalFeatures

    def validate(self) -> None:
        if len(self.eyes) not in (1, 2):
            raise ValueError("This blueprint expects 1 or 2 eye images per patient.")
        for eye in self.eyes:
            if not eye.image_path.exists():
                raise FileNotFoundError(f"Eye image not found: {eye.image_path}")


@dataclass
class WeibullParams:
    scale_lambda: float
    shape_p: float

    def cdf(self, t_months: float) -> float:
        import math
        return 1.0 - math.exp(-self.scale_lambda * (t_months ** self.shape_p))


@dataclass
class EyeRiskResult:
    eye_id: str
    p_no_dr: float
    p_any_dr: float
    image_risk_curve: dict[int, float] | dict[str, float]
    personalized_risk_curve: dict[int, float] | dict[str, float]
    eta_breakdown: Optional[dict[str, Any]] = None


@dataclass
class PatientRiskResult:
    patient_id: str
    clinical_multiplier: float
    eta_clinical: float
    eye_results: list[EyeRiskResult]
    patient_risk_curve: dict[int, float] | dict[str, float]
    suggested_followup_months: Optional[int]
    followup_threshold: float
    aggregation_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)