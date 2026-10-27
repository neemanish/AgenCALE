#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper (about 20 minutes on a 2-core CPU).
set -euo pipefail
OUT=${1:-results}
mkdir -p "$OUT"
python3 privenc_experiment.py "$OUT"   # 12 configurations x 3 seeds, attack suite, timing
python3 scaling.py "$OUT"              # scrubbing-loop cost at 2.5k-20k calibration notes
python3 analyze.py "$OUT"              # summary.csv, timing.csv, EPGI weighting, Figs. 2-3
python3 agencale.py "$OUT"             # agent coordinator, privacy gate, evidence logs
python3 fig_agents.py "$OUT"           # Fig. 1
echo "Done. See $OUT/"
