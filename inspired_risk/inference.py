from __future__ import annotations

from .clinical import compute_clinical_multiplier
from .config import FOLLOWUP_POLICY
from .curves import (
    aggregate_eye_curves,
    apply_clinical_multiplier,
    build_reference_weibulls,
    expected_image_risk_curve,
    suggest_followup,
)
from .schemas import EyeRiskResult, PatientInput, PatientRiskResult


class InspiredRiskPipeline:
    def __init__(self, image_model, aggregation_mode: str | None = None):
        self.image_model = image_model
        self.aggregation_mode = aggregation_mode or FOLLOWUP_POLICY["default_aggregation_mode"]
        self.ref_weibulls = build_reference_weibulls()

    def predict_patient(
        self,
        patient: PatientInput,
        followup_threshold: float | None = None,
    ) -> PatientRiskResult:
        patient.validate()
        followup_threshold = (
            followup_threshold
            if followup_threshold is not None
            else FOLLOWUP_POLICY["default_threshold"]
        )

        eye_results = []
        personalized_curves = []
        eta_values = []
        multipliers = []
        eta_breakdowns = []

        for eye in patient.eyes:
            p_no_dr, p_any_dr = self.image_model.predict_proba_from_path(eye.image_path)

            img_curve = expected_image_risk_curve(
                p_no_dr=p_no_dr,
                p_any_dr=p_any_dr,
                params_no_dr=self.ref_weibulls["no_dr"],
                params_any_dr=self.ref_weibulls["any_dr"],
            )

            eta, m_clin, eta_breakdown = compute_clinical_multiplier(
                patient.clinical,
                p_any_dr=p_any_dr,
            )

            personal_curve = apply_clinical_multiplier(img_curve, m_clin)

            eta_values.append(float(eta))
            multipliers.append(float(m_clin))
            eta_breakdowns.append(eta_breakdown)
            personalized_curves.append(personal_curve)

            eye_results.append(
                EyeRiskResult(
                    eye_id=eye.eye_id,
                    p_no_dr=float(p_no_dr),
                    p_any_dr=float(p_any_dr),
                    image_risk_curve=img_curve,
                    personalized_risk_curve=personal_curve,
                    eta_breakdown=eta_breakdown,
                )
            )

        patient_curve = aggregate_eye_curves(
            personalized_curves,
            mode=self.aggregation_mode,
        )
        followup = suggest_followup(
            patient_curve,
            threshold=followup_threshold,
        )

        # patient-level summary of eye-level clinical terms
        patient_eta = max(eta_values) if eta_values else 0.0
        patient_m_clin = max(multipliers) if multipliers else 1.0

        # choose the eye with the maximum multiplier as the representative summary
        representative_breakdown = None
        if eta_breakdowns:
            rep_idx = max(range(len(multipliers)), key=lambda i: multipliers[i])
            representative_breakdown = eta_breakdowns[rep_idx]

        metadata = {
            "note": (
                "Clinical multiplier is computed per eye because gender is weighted "
                "by p_any_dr in the blueprint."
            ),
            "patient_curve_definition": (
                f"Aggregation across eyes uses mode='{self.aggregation_mode}'."
            ),
            "config_design": (
                "All editable literature priors are centralized in inspired_risk/config.py."
            ),
        }

        if representative_breakdown is not None:
            metadata["clinical_ablation_mode"] = representative_breakdown.get("ablation_mode")
            metadata["clinical_alpha"] = representative_breakdown.get("alpha")
            metadata["clinical_eta_effective"] = representative_breakdown.get("eta_effective")

        return PatientRiskResult(
            patient_id=patient.patient_id,
            clinical_multiplier=float(patient_m_clin),
            eta_clinical=float(patient_eta),
            eye_results=eye_results,
            patient_risk_curve=patient_curve,
            suggested_followup_months=followup,
            followup_threshold=float(followup_threshold),
            aggregation_mode=self.aggregation_mode,
            metadata=metadata,
        )