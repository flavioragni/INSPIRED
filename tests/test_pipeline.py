from inspired_risk.clinical import compute_clinical_multiplier
from inspired_risk.config import CLINICAL_PRIORS, CURVE_CONFIG
from inspired_risk.curves import build_reference_weibulls, compute_any_dr_weighted_risks, expected_image_risk_curve
from inspired_risk.schemas import ClinicalFeatures


def test_weighted_any_dr_risk_is_reasonable():
    risks = compute_any_dr_weighted_risks()
    assert 0.02 < risks[12] < 0.04
    assert 0.09 < risks[36] < 0.12
    assert 0.16 < risks[60] < 0.19


def test_weibull_curves_are_monotonic():
    curves = build_reference_weibulls()
    vals = [curves["no_dr"].cdf(t) for t in [12, 36, 60]]
    assert vals[0] < vals[1] < vals[2]


def test_expected_curve_bounded():
    curves = build_reference_weibulls()
    out = expected_image_risk_curve(0.4, 0.6, curves["no_dr"], curves["any_dr"])
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_clinical_multiplier_positive():
    clin = ClinicalFeatures(
        diabetes_type="T2D",
        diabetes_duration_years=15,
        hba1c=8.0,
        systolic_bp=145,
        sex="male",
        egfr=70,
        fenofibrate=False,
        anemia=True,
    )
    eta, m = compute_clinical_multiplier(clin, p_any_dr=0.7)
    assert eta > 0
    assert m > 1


def test_all_editable_priors_are_centralized():
    assert "hba1c" in CLINICAL_PRIORS
    assert "sbp" in CLINICAL_PRIORS
    assert "egfr" in CLINICAL_PRIORS
    assert "fenofibrate" in CLINICAL_PRIORS
    assert "anemia" in CLINICAL_PRIORS
    assert "sex_by_type_if_dr" in CLINICAL_PRIORS
    assert "duration_by_type" in CLINICAL_PRIORS
    assert "no_dr" in CURVE_CONFIG
    assert "any_dr" in CURVE_CONFIG
