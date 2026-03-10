#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
make clean
make
./simulation --steps 500
