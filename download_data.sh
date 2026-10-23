#!/usr/bin/env bash
# Download the cropped CREMI volumes (training A, B, C with labels; test A+, B+, C+) into data/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
for f in sample_A_20160501 sample_B_20160501 sample_C_20160501 \
         sample_A+_20160601 sample_B+_20160601 sample_C+_20160601; do
  [ -f "data/$f.hdf" ] && { echo "have $f"; continue; }
  echo "downloading $f"
  curl -fL --progress-bar -o "data/$f.hdf" "https://cremi.org/static/data/${f//+/%2B}.hdf"
done
