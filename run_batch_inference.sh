#!/usr/bin/env bash

set -euo pipefail

MODE="full_alpha0.5"
PATIENT_DIR="path/to/patients/folder"
OUTPUT_DIR="path/to/output/dir/$MODE"
OUTPUT_PLOTS_DIR="/path/to/plots/dir/$MODE"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_PLOTS_DIR"

for patient_json in "$PATIENT_DIR"/*.json; do
    [ -e "$patient_json" ] || continue

    output_json="$OUTPUT_DIR/$(basename "$patient_json")"

    python run_inference.py \
        --patient-json "$patient_json" \
        --checkpoint /path/to/checkpoint \
        --model-config-json path/to/config \
        --temperature-checkpoint /path/to/temp \
        --device cpu \
        --aggregation-mode max \
        --followup-threshold 0.03 \
        --output-json "$output_json"
done

# Make plots
python plots/plot_inspired_results.py \
  --predictions-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_PLOTS_DIR"
