# INSPIRED Retinal Risk Model

Research prototype for estimating personalised diabetic-retinopathy progression risk from retinal fundus images and clinical risk factors.

The pipeline combines:

* a calibrated retinal-image classifier;
* literature-informed Weibull reference risk curves;
* clinical hazard adjustments;
* one- or two-eye risk aggregation;
* a prototype follow-up recommendation policy;
* cohort-level diagnostic and interpretability plots.

> **Research use only:** this repository is an experimental prototype. Its risk estimates and follow-up recommendations have not been prospectively validated and must not be used for clinical decision-making.

## Overview

```text
Patient retinal images and clinical variables
                       │
                       ▼
             Retinal image classifier
                       │
             p(no DR) and p(any DR)
                       │
                       ▼
             Weibull reference curves
                       │
                       ▼
            Image-based expected risk
                       │
                       ▼
            Clinical hazard adjustment
                       │
                       ▼
          One- or two-eye risk aggregation
                       │
                       ▼
          Personalised cumulative-risk curve
                       │
                       ▼
       Prototype suggested follow-up interval
```

## Repository structure

```text
.
├── code/
│   ├── preprocessing/
│   │   ├── create_patients_json.ipynb
│   │   └── questionario_df_mod.csv
│   │
│   ├── inference/
│   │   ├── inspired_risk/
│   │   │   ├── __init__.py
│   │   │   ├── clinical.py
│   │   │   ├── config.py
│   │   │   ├── curves.py
│   │   │   ├── inference.py
│   │   │   ├── io_utils.py
│   │   │   ├── modeling.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── checkpoints/
│   │   │   ├── unfreeze_all.pth
│   │   │   ├── unfreeze_all.json
│   │   │   └── temperature_scaler.pth
│   │   │
│   │   ├── examples/
│   │   │   ├── left_eye.png
│   │   │   ├── right_eye.png
│   │   │   ├── patient_demo.json
│   │   │   ├── prediction_demo.json
│   │   │   └── prediction_dummy.json
│   │   │
│   │   ├── patients_json/
│   │   │   ├── image_list_clean.csv
│   │   │   └── <patient_id>.json
│   │   │
│   │   ├── predictions_json/
│   │   │   ├── full/
│   │   │   ├── full_alpha0.5/
│   │   │   ├── full_alpha1/
│   │   │   ├── full_noegfr/
│   │   │   └── ...
│   │   │
│   │   ├── plots/
│   │   │   ├── plot_inspired_results.py
│   │   │   └── ...
│   │   │
│   │   ├── tests/
│   │   │   └── test_pipeline.py
│   │   │
│   │   ├── create_demo_inputs.py
│   │   ├── create_dummy_checkpoint.py
│   │   ├── run_inference.py
│   │   ├── run_batch_inference.sh
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   └── fit_weibull.py
│
├── .gitignore
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

The inference pipeline requires:

```text
numpy
scipy
pillow
torch
torchvision
opencv-python
pandas
matplotlib
openpyxl
jupyter
pytest
```

Install the dependencies with:

```bash
pip install \
  numpy \
  scipy \
  pillow \
  torch \
  torchvision \
  opencv-python \
  pandas \
  matplotlib \
  openpyxl \
  jupyter \
  pytest
```

Install PyTorch and torchvision using versions appropriate for the local CPU or CUDA environment.

## Patient input

The model accepts one JSON file per patient.

Each file contains:

* a pseudonymous patient identifier;
* one or two retinal image paths;
* clinical risk variables.

### Input format

```json
{
  "patient_id": "example_patient",
  "eyes": [
    {
      "eye_id": "l",
      "image_path": "/absolute/path/to/left_eye.jpg"
    },
    {
      "eye_id": "r",
      "image_path": "/absolute/path/to/right_eye.jpg"
    }
  ],
  "clinical": {
    "diabetes_type": "T2D",
    "diabetes_duration_years": 15.0,
    "hba1c": 7.1,
    "systolic_bp": 135.0,
    "sex": "male",
    "egfr": 75.0,
    "fenofibrate": false,
    "anemia": false
  }
}
```

The pipeline accepts one or two eyes. Every referenced image path must exist.

## Clinical variables

The current model supports:

| Variable                  | Description                           |
| ------------------------- | ------------------------------------- |
| `diabetes_type`           | Diabetes type, such as `T1D` or `T2D` |
| `diabetes_duration_years` | Duration of diabetes in years         |
| `hba1c`                   | Glycated haemoglobin                  |
| `systolic_bp`             | Systolic blood pressure in mmHg       |
| `sex`                     | Patient sex                           |
| `egfr`                    | Estimated glomerular filtration rate  |
| `fenofibrate`             | Fenofibrate treatment status          |
| `anemia`                  | Anaemia status                        |

Missing clinical values are accepted and contribute a neutral adjustment for the corresponding term.

## Creating patient JSON files

The notebook:

```text
code/preprocessing/create_patients_json.ipynb
```

creates patient-level JSON inputs by combining:

* selected retinal image paths;
* questionnaire information;
* laboratory values;
* anamnesis data;
* pharmacological treatment data.

Open it with:

```bash
jupyter lab code/preprocessing/create_patients_json.ipynb
```

The notebook currently reads external tables such as:

```text
image_list_clean.csv
esami_lab_df.csv
anamnesi_df.csv
questionario_df_mod.csv
farmacologia_altro_df.csv
```

Update the absolute paths before execution.

### Clinical-variable mapping

The notebook derives:

| Output variable           | Source                                       |
| ------------------------- | -------------------------------------------- |
| `diabetes_type`           | Diabetes type from the anamnesis table       |
| `diabetes_duration_years` | Corrected diabetes duration                  |
| `hba1c`                   | HbA1c measured in the image-acquisition year |
| `systolic_bp`             | Systolic blood pressure from the same year   |
| `sex`                     | Questionnaire data                           |
| `egfr`                    | CKD-EPI eGFR from the same year              |
| `fenofibrate`             | Fenofibrate treatment during the same year   |
| `anemia`                  | Questionnaire data                           |

The image-acquisition year is extracted from the retinal-image filename.

### Shallow-copy issue

The notebook currently constructs patient dictionaries using a shallow copy:

```python
patient_data = patient_json.copy()
```

Because nested lists and dictionaries can be shared across patients, use:

```python
from copy import deepcopy

patient_data = deepcopy(patient_json)
```

or create a completely new dictionary for every patient.

## Retinal image classifier

The classifier is implemented in:

```text
code/inference/inspired_risk/modeling.py
```

The current architecture uses a torchvision ResNet-50 backbone.

### Image preprocessing

Each fundus image undergoes:

1. RGB conversion;
2. CLAHE contrast enhancement in LAB colour space;
3. resize to 256 pixels;
4. centre crop to 224 × 224;
5. conversion to a PyTorch tensor;
6. ImageNet normalisation.

The normalisation values are:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### Model output

The network produces one binary logit.

The output probabilities are:

```text
p_any_dr = sigmoid(logit)
p_no_dr = 1 - p_any_dr
```

The model therefore estimates the probability that the image contains any diabetic retinopathy rather than no diabetic retinopathy.

## Reference risk curves

Reference cumulative-risk curves are implemented in:

```text
code/inference/inspired_risk/curves.py
```

The pipeline uses Weibull curves:

```text
R(t) = 1 - exp(-λtᵖ)
```

where:

* `t` is time;
* `λ` controls scale;
* `p` controls shape.

The parameters are fitted numerically from configured cumulative-risk anchors.

### No-DR anchors

|   Horizon | Cumulative risk |
| --------: | --------------: |
| 12 months |            0.3% |
| 36 months |            1.0% |
| 60 months |            2.2% |

### Any-DR anchors

The Any-DR curve combines five severity groups:

```text
very mild
mild
moderate
severe
very severe
```

using proportions configured in:

```text
code/inference/inspired_risk/config.py
```

The resulting weighted risks are approximately:

|   Horizon | Cumulative risk |
| --------: | --------------: |
| 12 months |            2.8% |
| 36 months |           10.6% |
| 60 months |           17.8% |

Some severity-specific intermediate values were introduced as monotonic prototype assumptions. All anchors must be verified against the intended literature before clinical interpretation.

## Image-based risk

The classifier probabilities are used to combine the No-DR and Any-DR reference curves:

```text
Rimage(t) =
    p(no DR) × Rno-DR(t)
    +
    p(any DR) × Rany-DR(t)
```

The default prediction horizons are:

```text
6, 12, 24, 36, 48, and 60 months
```

## Clinical adjustment

Clinical adjustments are implemented in:

```text
code/inference/inspired_risk/clinical.py
```

and configured in:

```text
code/inference/inspired_risk/config.py
```

The clinical predictor can include:

* diabetes-duration category;
* HbA1c relative to 7%;
* systolic blood pressure relative to 130 mmHg;
* eGFR category for type 2 diabetes;
* fenofibrate treatment;
* anaemia for type 2 diabetes;
* sex and diabetes-type interaction weighted by `p_any_dr`.

The raw clinical predictor is:

```text
ηclinical = Σ ηterm
```

The clinical multiplier is:

```text
Mclinical = exp(α × ηclinical)
```

The default value is:

```text
α = 0.5
```

The personalised risk is calculated by scaling cumulative hazard:

```text
Rpersonalised(t) =
    1 - [1 - Rimage(t)]^Mclinical
```

Because some clinical terms depend on the eye-level `p_any_dr`, the multiplier can differ between the two eyes.

## Clinical ablation modes

The following modes are supported:

| Mode             | Included variables               |
| ---------------- | -------------------------------- |
| `none`           | Image risk only                  |
| `duration_only`  | Diabetes duration                |
| `core`           | Duration, HbA1c, and systolic BP |
| `core_plus_egfr` | Core variables and eGFR          |
| `full_no_sex`    | All terms except sex             |
| `full_no_egfr`   | All terms except eGFR            |
| `full`           | All configured variables         |

The mode is currently set in source code:

```python
CLINICAL_ABLATION_CONFIG = {
    "mode": "full"
}
```

The effect scaling is configured with:

```python
CLINICAL_MULTIPLIER_CONFIG = {
    "alpha": 0.5,
    "max_multiplier": None
}
```

These parameters are not currently exposed through the command-line interface.

## Follow-up recommendation

The prototype follow-up policy uses a default cumulative-risk threshold of:

```text
3%
```

Risk is evaluated at:

```text
6, 12, 24, 36, 48, and 60 months
```

The algorithm returns the horizon immediately before the first threshold crossing.

| First horizon above 3% | Suggested follow-up |
| ---------------------: | ------------------: |
|               6 months |           Immediate |
|              12 months |            6 months |
|              24 months |           12 months |
|              36 months |           24 months |
|              48 months |           36 months |
|              60 months |           48 months |
|            No crossing |           60 months |

This policy is experimental and has not been clinically validated.

## Run inference for one patient

Change to the inference directory:

```bash
cd code/inference
```

Run:

```bash
python run_inference.py \
  --patient-json patients_json/example_patient.json \
  --checkpoint checkpoints/unfreeze_all.pth \
  --model-config-json checkpoints/unfreeze_all.json \
  --temperature-checkpoint checkpoints/temperature_scaler.pth \
  --device cpu \
  --aggregation-mode max \
  --followup-threshold 0.03 \
  --output-json prediction.json
```

### Arguments

| Argument                   | Description                             |
| -------------------------- | --------------------------------------- |
| `--patient-json`           | Patient input JSON                      |
| `--checkpoint`             | Classifier checkpoint                   |
| `--model-config-json`      | Model architecture configuration        |
| `--temperature-checkpoint` | Optional temperature-scaling checkpoint |
| `--device`                 | PyTorch device such as `cpu` or `cuda`  |
| `--aggregation-mode`       | `max`, `mean`, or `union_independent`   |
| `--followup-threshold`     | Cumulative-risk threshold               |
| `--output-json`            | Output JSON destination                 |

## Prediction output

The output JSON contains:

```text
patient_id
clinical_multiplier
eta_clinical
eye_results
patient_risk_curve
suggested_followup_months
followup_threshold
aggregation_mode
metadata
```

Each eye contains:

```text
eye_id
p_no_dr
p_any_dr
image_risk_curve
personalized_risk_curve
eta_breakdown
```

The clinical contribution breakdown can contain:

```text
duration
hba1c
sbp
egfr
fenofibrate
anemia
sex
eta_raw
alpha
eta_effective
m_clin
ablation_mode
```

Example:

```json
{
  "patient_id": "example_patient",
  "clinical_multiplier": 1.52,
  "eta_clinical": 0.84,
  "eye_results": [
    {
      "eye_id": "l",
      "p_no_dr": 0.83,
      "p_any_dr": 0.17,
      "image_risk_curve": {
        "6": 0.0043,
        "12": 0.009,
        "24": 0.0187
      },
      "personalized_risk_curve": {
        "6": 0.0065,
        "12": 0.0136,
        "24": 0.0282
      },
      "eta_breakdown": {
        "duration": 0.72,
        "hba1c": 0.03,
        "eta_raw": 0.83,
        "alpha": 0.5,
        "eta_effective": 0.41,
        "m_clin": 1.51,
        "ablation_mode": "full"
      }
    }
  ],
  "patient_risk_curve": {
    "6": 0.0065,
    "12": 0.0136,
    "24": 0.0282
  },
  "suggested_followup_months": 24,
  "followup_threshold": 0.03,
  "aggregation_mode": "max"
}
```

## Cohort-level diagnostics

Generate cohort plots with:

```bash
cd code/inference

python plots/plot_inspired_results.py \
  --predictions-dir predictions_json/full_alpha0.5 \
  --output-dir plots/full_alpha0.5
```

The plotting workflow can generate:

```text
figure1_followup_distribution.png
figure2_patient_risk_spaghetti.png
figure3_image_vs_personalized_scatter_24m.png
figure4_transition_heatmap.png
figure5_waterfall_patient_decomposition_24m.png
figure6_image_vs_final_risk_mclin_24m.png
mclin_histogram.png
eta_contribution_boxplots.png
eta_contribution_mean_abs.png
patient_eta_contribution_matrix.png
patient_eta_matrix_row_lookup.csv
clinical_diagnostic_summary.json
eta_contribution_summary.csv
```

These outputs visualise:

* suggested follow-up intervals;
* patient cumulative-risk trajectories;
* image-only versus clinically adjusted risk;
* follow-up-category changes after clinical adjustment;
* per-patient risk decomposition;
* distributions of clinical multipliers;
* variable-level clinical contributions;
* patient-by-variable contribution matrices.

## Citation

Add the associated INSPIRED publication when available:

```bibtex
@article{author_year_inspired,
  title   = {Title of the associated INSPIRED study},
  author  = {Author list},
  journal = {Journal},
  year    = {Year},
  doi     = {DOI}
}
```
