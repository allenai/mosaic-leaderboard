#!/usr/bin/env bash

set -e

python random_baseline.py --input-file /data/dev.jsonl --output-file /results/predictions.lst
