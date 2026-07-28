from __future__ import annotations

DEFAULT_HORIZONS_MONTHS = [6, 12, 24, 36, 48, 60]

FOLLOWUP_POLICY = {
    "default_threshold": 0.03,
    "default_aggregation_mode": "max",
}

CLINICAL_MULTIPLIER_CONFIG = {
    "alpha": 0.5,              # 1.0 = full literature effect
    "max_multiplier": None,    # e.g. 3.0 if you want a hard cap
}

CLINICAL_ABLATION_CONFIG = {
    "mode": "full",
    "allowed_modes": [
        "none",            # image only
        "duration_only",
        "core",            # duration + HbA1c + SBP
        "core_plus_egfr",  # core + eGFR
        "full_no_sex",     # full except sex
        "full_no_egfr",    # full except eGFR
        "full",            # all enabled terms
    ],
}

CURVE_CONFIG = {
    "no_dr": {
        "time_to_risk": {
            12: 0.003,
            36: 0.010,
            60: 0.022,
        }
    },
    "any_dr": {
        "severity_group_risks": {
            "very_mild": {12: 0.020, 36: 0.070, 60: 0.079},
            "mild": {12: 0.022, 36: 0.085, 60: 0.130},
            "moderate": {12: 0.040, 36: 0.160, 60: 0.272},
            "severe": {12: 0.090, 36: 0.280, 60: 0.455},
            "very_severe": {12: 0.180, 36: 0.450, 60: 0.677},
        },
        "severity_weights": {
            "very_mild": 0.3697,
            "mild": 0.2956,
            "moderate": 0.2508,
            "severe": 0.0649,
            "very_severe": 0.0189,
        },
    },
}

CLINICAL_PRIORS = {
    "duration_by_type": {
        "T1D": {
            "<10": {"min_years": 0.0, "max_years": 10.0, "rr": 1.38},
            "10-<20": {"min_years": 10.0, "max_years": 20.0, "rr": 2.43},
            "20+": {"min_years": 20.0, "max_years": float("inf"), "rr": 2.69},
        },
        "T2D": {
            "<10": {"min_years": 0.0, "max_years": 10.0, "rr": 1.00},
            "10-<20": {"min_years": 10.0, "max_years": 20.0, "rr": 2.06},
            "20+": {"min_years": 20.0, "max_years": float("inf"), "rr": 2.45},
        },
    },
    "hba1c": {
        "rr_per_1pct": 1.29,
        "reference": 7.0,
        "unit_step": 1.0,
    },
    "sbp": {
        "hr_per_unit": 1.24,
        "reference": 130.0,
        "unit_step": 10.0,
    },
    "egfr": {
        "categories": {
            ">90": {"min": 90.0, "max": float("inf"), "hr": 1.0},
            "60-89": {"min": 60.0, "max": 90.0, "hr": 1.649},
            "<60": {"min": float("-inf"), "max": 60.0, "hr": 2.106},
        },
        "applies_only_to": ["T2D"],
    },
    "fenofibrate": {
        "hr_if_yes": 0.72,
    },
    "anemia": {
        "hr_if_yes": 1.29,
        "applies_only_to": ["T2D"],
    },
    "sex_by_type_if_dr": {
        "T1D": {"female": -0.194, "male": 0.194},
        "T2D": {"female": -0.46, "male": 0.46},
        "interpretation": "Aspelund beta_DR coefficients on the linear predictor scale, applied as beta_DR * p_any_dr.",
    },
    "bilirubin": {
        "enabled": False,
        "note": "Blueprint marks bilirubin as uncertain; left disabled until finalized.",
    },
}