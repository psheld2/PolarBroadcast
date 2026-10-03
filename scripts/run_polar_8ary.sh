#!/usr/bin/env bash
# All polar-simulation jobs for the report (n = 128, 13/8 dB), several in parallel.
# Usage (on the compute host, from the scripts folder): bash run_polar_8ary.sh [parallel_jobs]
set -euo pipefail
P=${1:-14}
PY=${PY:-~/venvs/sim/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
for kind in 8psk 8qam; do
  for rm in short punct; do
    for mu in 0 0.25 0.5 0.75 1; do echo "$kind mix $mu --rm $rm"; done
    echo "$kind tdm 0 1 --rm $rm"; echo "$kind tdm 1 1 --rm $rm"
    for lam in 0.25 0.5 0.75; do for p1 in 0.6 1.0 1.3; do echo "$kind tdm $lam $p1 --rm $rm"; done; done
  done
  for a in 0.02 0.05 0.1 0.2; do echo "$kind gag $a"; done
done
} | xargs -P "$P" -I{} bash -c "$PY polar_8ary_sim.py {} >> polar_8ary_sim.log 2>&1 || echo 'FAILED: {}' >> polar_8ary_sim.log"
echo ALL-DONE >> polar_8ary_sim.log
