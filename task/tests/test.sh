#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier

for required in /tests/test_outputs.py /tests/reference/expected.json; do
  if [ ! -f "$required" ]; then
    echo "FATAL: verifier bundle missing required file: $required" >&2
    echo 0 > /logs/verifier/reward.txt
    exit 1
  fi
done

set +e
python3 -m pytest \
    -o cache_dir=/tmp/pytest_cache \
    --ctrf /logs/verifier/ctrf.json \
    /tests/test_outputs.py -rA
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit "$RC"
