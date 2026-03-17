#!/bin/bash
set -euo pipefail
cd /app
export OMP_NUM_THREADS=4
make clean
make
/app/simulation --steps 500
