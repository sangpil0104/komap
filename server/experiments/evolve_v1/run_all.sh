#!/bin/bash
# Sequential, single-GPU: trains and evaluates all 100 evolve_v1 experiments one at a time.
# Progress: terminal tqdm + outputs/suite_status.json only (no localhost server/dashboard).
exec conda run --no-capture-output -n komap python -B run.py suite
