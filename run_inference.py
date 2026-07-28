from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from inspired_risk.inference import InspiredRiskPipeline
from inspired_risk.io_utils import load_patient_json
from inspired_risk.modeling import TorchDRClassifier

###
# python run_inference.py \
#   --patient-json examples/patient_demo.json \
#   --checkpoint checkpoints/best_model.pt \
#   --model-config-json checkpoints/model_config.json \
#   --temperature-checkpoint checkpoints/temperature_scaler.pt \
#   --device cpu \
#   --aggregation-mode max \
#   --followup-threshold 0.03 \
#   --output-json examples/prediction_demo.json
###

def main() -> None:
    parser = argparse.ArgumentParser(description="Run INSPIRED risk prediction for one patient.")
    parser.add_argument("--patient-json", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config-json", type=Path, default=None, help="JSON file with model hyperparameters used to instantiate OldInspiredModel.")
    parser.add_argument("--temperature-checkpoint", type=Path, default=None, help="Optional checkpoint for TemperatureScaler.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--followup-threshold", type=float, default=0.03)
    parser.add_argument("--aggregation-mode", choices=["max", "mean", "union_independent"], default="max")
    parser.add_argument("--output-json", type=Path, default=Path("prediction.json"))
    args = parser.parse_args()

    patient = load_patient_json(args.patient_json)
    image_model = TorchDRClassifier(
        checkpoint_path=args.checkpoint,
        temperature_checkpoint_path=args.temperature_checkpoint,
        model_config_path=args.model_config_json,
        device=args.device,
    )
    pipeline = InspiredRiskPipeline(image_model=image_model, aggregation_mode=args.aggregation_mode)
    result = pipeline.predict_patient(patient, followup_threshold=args.followup_threshold)

    args.output_json.write_text(json.dumps(asdict(result), indent=2, default=str))
    print(json.dumps(asdict(result), indent=2, default=str))
    print(f"\nSaved output to {args.output_json}")


if __name__ == "__main__":
    main()