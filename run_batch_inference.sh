#!/usr/bin/env bash

set -euo pipefail

MODE="full_alpha0.5"
PATIENT_DIR="/storage/DSH/projects/inspired/flavio/inspired/code/inference/patients_json"
OUTPUT_DIR="/storage/DSH/projects/inspired/flavio/inspired/code/inference/predictions_json/$MODE"
OUTPUT_PLOTS_DIR="/storage/DSH/projects/inspired/flavio/inspired/code/inference/plots/$MODE"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_PLOTS_DIR"

for patient_json in "$PATIENT_DIR"/*.json; do
    [ -e "$patient_json" ] || continue

    output_json="$OUTPUT_DIR/$(basename "$patient_json")"

    python run_inference.py \
        --patient-json "$patient_json" \
        --checkpoint /storage/DSH/projects/inspired/flavio/inspired/code/inference/checkpoints/unfreeze_all.pth \
        --model-config-json /storage/DSH/projects/inspired/flavio/inspired/code/inference/checkpoints/unfreeze_all.json \
        --temperature-checkpoint /storage/DSH/projects/inspired/flavio/inspired/code/inference/checkpoints/temperature_scaler.pth \
        --device cpu \
        --aggregation-mode max \
        --followup-threshold 0.03 \
        --output-json "$output_json"
done

# Make plots
python plots/plot_inspired_results.py \
  --predictions-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_PLOTS_DIR"