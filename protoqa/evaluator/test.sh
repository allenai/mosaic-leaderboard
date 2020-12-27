#!/bin/bash

set -xe

docker build -t protoqa-leaderboard-eval-test .

T=$(mktemp -d /tmp/tmp-XXXXX)

docker run \
  -v $T:/output:rw \
  -v $PWD:/input:ro \
  protoqa-leaderboard-eval-test \
  python \
  eval.py \
  --labels_file /input/labels.jsonl \
  --preds_file /input/predictions.jsonl \
  --metrics_output_file /output/metrics.json

cat $T/metrics.json