from __future__ import annotations

import math

from .config import (
    CLINICAL_ABLATION_CONFIG,
    CLINICAL_MULTIPLIER_CONFIG,
    CLINICAL_PRIORS,
)
from .schemas import ClinicalFeatures


ETA_TERM_ORDER = [
    "duration",
    "hba1c",
    "sbp",
    "egfr",
    "fenofibrate",
    "anemia",
    "sex",
]


def _normalize_scalar(x):
    """
    Normalize optional scalar inputs.

    Returns None for:
    - None
    - empty list
    - single-item list containing None/NaN
    - NaN numeric values
    - empty strings
    - common missing string tokens
    """
    if x is None:
        return None

    if isinstance(x, list):
        if len(x) == 0:
            return None
        if len(x) == 1:
            return _normalize_scalar(x[0])
        raise ValueError(f"Expected scalar value, got list: {x}")

    if isinstance(x, str):
        x_str = x.strip()
        if x_str == "" or x_str.lower() in {"nan", "none", "null", "na", "n/a"}:
            return None
        return x_str

    try:
        x_float = float(x)
        if math.isnan(x_float):
            return None
    except (TypeError, ValueError):
        pass

    return x


def _duration_rr(diabetes_type: str, years: float) -> float:
    bins = CLINICAL_PRIORS["duration_by_type"][diabetes_type]
    for spec in bins.values():
        if spec["min_years"] <= years < spec["max_years"]:
            return float(spec["rr"])
    raise ValueError(f"No duration RR found for type={diabetes_type}, years={years}")


def _egfr_hr(egfr: float) -> float:
    """
    Return eGFR hazard ratio.

    Missing/NaN eGFR should be neutral, not treated as <60.
    """
    if egfr is None or math.isnan(float(egfr)):
        return 1.0

    categories = CLINICAL_PRIORS["egfr"]["categories"]

    if egfr >= categories[">90"]["min"]:
        return float(categories[">90"]["hr"])

    if categories["60-89"]["min"] <= egfr < categories["60-89"]["max"]:
        return float(categories["60-89"]["hr"])

    return float(categories["<60"]["hr"])


def _get_ablation_mode() -> str:
    mode = CLINICAL_ABLATION_CONFIG.get("mode", "full")
    allowed = set(CLINICAL_ABLATION_CONFIG.get("allowed_modes", []))
    if allowed and mode not in allowed:
        raise ValueError(f"Unsupported clinical ablation mode: {mode}")
    return mode


def _term_enabled(term_name: str, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "duration_only":
        return term_name == "duration"
    if mode == "core":
        return term_name in {"duration", "hba1c", "sbp"}
    if mode == "core_plus_egfr":
        return term_name in {"duration", "hba1c", "sbp", "egfr"}
    if mode == "full_no_sex":
        return term_name in {"duration", "hba1c", "sbp", "egfr", "fenofibrate", "anemia"}
    if mode == "full_no_egfr":
        return term_name in {"duration", "hba1c", "sbp", "fenofibrate", "anemia", "sex"}
    if mode == "full":
        return True
    raise ValueError(f"Unsupported clinical ablation mode: {mode}")


def decompose_eta_clinical(clin: ClinicalFeatures, p_any_dr: float) -> dict[str, float]:
    priors = CLINICAL_PRIORS
    mode = _get_ablation_mode()

    diabetes_type = _normalize_scalar(clin.diabetes_type)
    diabetes_duration_years = _normalize_scalar(clin.diabetes_duration_years)
    hba1c = _normalize_scalar(clin.hba1c)
    systolic_bp = _normalize_scalar(clin.systolic_bp)
    egfr = _normalize_scalar(clin.egfr)
    fenofibrate = _normalize_scalar(clin.fenofibrate)
    anemia = _normalize_scalar(clin.anemia)
    sex = _normalize_scalar(clin.sex)

    if isinstance(diabetes_type, str):
        diabetes_type = diabetes_type.strip()

    if isinstance(sex, str):
        sex = sex.strip().lower()

    contrib = {k: 0.0 for k in ETA_TERM_ORDER}

    # Duration
    if _term_enabled("duration", mode):
        if diabetes_type is not None and diabetes_duration_years is not None:
            contrib["duration"] = math.log(
                _duration_rr(diabetes_type, float(diabetes_duration_years))
            )

    # HbA1c
    if _term_enabled("hba1c", mode):
        if hba1c is not None:
            cfg = priors["hba1c"]
            contrib["hba1c"] = math.log(cfg["rr_per_1pct"]) * (
                (float(hba1c) - cfg["reference"]) / cfg["unit_step"]
            )

    # SBP
    if _term_enabled("sbp", mode):
        if systolic_bp is not None:
            cfg = priors["sbp"]
            contrib["sbp"] = math.log(cfg["hr_per_unit"]) * (
                (float(systolic_bp) - cfg["reference"]) / cfg["unit_step"]
            )

    # eGFR
    if _term_enabled("egfr", mode):
        if (
            egfr is not None
            and diabetes_type in priors["egfr"]["applies_only_to"]
        ):
            contrib["egfr"] = math.log(_egfr_hr(float(egfr)))

    # Fenofibrate
    if _term_enabled("fenofibrate", mode):
        if fenofibrate is not None and bool(fenofibrate):
            contrib["fenofibrate"] = math.log(priors["fenofibrate"]["hr_if_yes"])

    # Anemia
    if _term_enabled("anemia", mode):
        if (
            anemia is not None
            and bool(anemia)
            and diabetes_type in priors["anemia"]["applies_only_to"]
        ):
            contrib["anemia"] = math.log(priors["anemia"]["hr_if_yes"])

    # Sex: Aspelund-style beta_DR modifier of DR presence.
    # This is not a standalone RR term.
    if _term_enabled("sex", mode):
        if diabetes_type is not None and sex is not None:
            sex_cfg = priors["sex_by_type_if_dr"]
            beta_dr = float(sex_cfg[diabetes_type][sex])
            contrib["sex"] = beta_dr * float(p_any_dr)

    return contrib


def compute_eta_clinical(clin: ClinicalFeatures, p_any_dr: float) -> float:
    contrib = decompose_eta_clinical(clin=clin, p_any_dr=p_any_dr)
    return sum(contrib.values())


def compute_clinical_multiplier(
    clin: ClinicalFeatures,
    p_any_dr: float,
) -> tuple[float, float, dict[str, float]]:
    contrib = decompose_eta_clinical(clin=clin, p_any_dr=p_any_dr)
    eta = sum(contrib.values())

    alpha = float(CLINICAL_MULTIPLIER_CONFIG.get("alpha", 1.0))
    eta_effective = alpha * eta
    m_clin = math.exp(eta_effective)

    max_multiplier = CLINICAL_MULTIPLIER_CONFIG.get("max_multiplier")
    if max_multiplier is not None:
        m_clin = min(m_clin, float(max_multiplier))

    contrib_out = dict(contrib)
    contrib_out["eta_raw"] = eta
    contrib_out["alpha"] = alpha
    contrib_out["eta_effective"] = eta_effective
    contrib_out["m_clin"] = m_clin
    contrib_out["ablation_mode"] = _get_ablation_mode()

    return eta, m_clin, contrib_out