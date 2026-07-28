from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .config import CURVE_CONFIG, DEFAULT_HORIZONS_MONTHS
from .schemas import WeibullParams


def _fit_weibull_from_points(time_to_risk: dict[int, float]) -> WeibullParams:
    times = np.array(sorted(time_to_risk.keys()), dtype=float)
    probs = np.array([time_to_risk[int(t)] for t in times], dtype=float)
    eps = 1e-8
    probs = np.clip(probs, eps, 1 - eps)

    def objective(theta: np.ndarray) -> float:
        log_lambda, log_p = theta
        lam = math.exp(log_lambda)
        p = math.exp(log_p)
        pred = 1.0 - np.exp(-lam * (times ** p))
        return float(np.sum((pred - probs) ** 2))

    y = np.log(-np.log(1.0 - probs))
    x = np.log(times)
    slope, intercept = np.polyfit(x, y, deg=1)
    init = np.array([intercept, np.log(max(slope, 1e-3))])

    result = minimize(objective, init, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Weibull fit failed: {result.message}")

    lam = math.exp(result.x[0])
    p = math.exp(result.x[1])
    return WeibullParams(scale_lambda=lam, shape_p=p)


def compute_any_dr_weighted_risks() -> dict[int, float]:
    any_dr_cfg = CURVE_CONFIG["any_dr"]
    group_risks = any_dr_cfg["severity_group_risks"]
    weights = any_dr_cfg["severity_weights"]
    horizons = sorted(next(iter(group_risks.values())).keys())
    out: dict[int, float] = {}
    for t in horizons:
        out[t] = sum(weights[g] * group_risks[g][t] for g in group_risks)
    return out


def build_reference_weibulls() -> dict[str, WeibullParams]:
    any_dr_risks = compute_any_dr_weighted_risks()
    return {
        "no_dr": _fit_weibull_from_points(CURVE_CONFIG["no_dr"]["time_to_risk"]),
        "any_dr": _fit_weibull_from_points(any_dr_risks),
    }


def evaluate_weibull_curve(params: WeibullParams, horizons_months: Iterable[int] = DEFAULT_HORIZONS_MONTHS) -> dict[int, float]:
    return {int(t): params.cdf(float(t)) for t in horizons_months}


def expected_image_risk_curve(
    p_no_dr: float,
    p_any_dr: float,
    params_no_dr: WeibullParams,
    params_any_dr: WeibullParams,
    horizons_months: Iterable[int] = DEFAULT_HORIZONS_MONTHS,
) -> dict[int, float]:
    if abs((p_no_dr + p_any_dr) - 1.0) > 1e-4:
        total = p_no_dr + p_any_dr
        if total <= 0:
            raise ValueError("Probabilities must sum to a positive value.")
        p_no_dr, p_any_dr = p_no_dr / total, p_any_dr / total

    curve: dict[int, float] = {}
    for t in horizons_months:
        r0 = params_no_dr.cdf(float(t))
        r1 = params_any_dr.cdf(float(t))
        curve[int(t)] = p_no_dr * r0 + p_any_dr * r1
    return curve


def apply_clinical_multiplier(image_curve: dict[int, float], m_clin: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for t, risk in image_curve.items():
        survival = max(1e-12, 1.0 - risk)
        out[int(t)] = 1.0 - (survival ** m_clin)
    return out


def aggregate_eye_curves(eye_curves: list[dict[int, float]], mode: str = "max") -> dict[int, float]:
    if not eye_curves:
        raise ValueError("At least one eye curve is required.")
    horizons = sorted(eye_curves[0].keys())

    if mode == "max":
        return {t: max(curve[t] for curve in eye_curves) for t in horizons}
    if mode == "mean":
        return {t: float(np.mean([curve[t] for curve in eye_curves])) for t in horizons}
    if mode == "union_independent":
        out = {}
        for t in horizons:
            surv = np.prod([1.0 - curve[t] for curve in eye_curves])
            out[t] = float(1.0 - surv)
        return out
    raise ValueError(f"Unknown aggregation mode: {mode}")


def suggest_followup(curve: dict[int, float], threshold: float) -> int | None:
    horizons = sorted(curve.keys())
    for idx, t in enumerate(horizons):
        if curve[t] > threshold:
            return horizons[idx - 1] if idx > 0 else 0
    return horizons[-1]
