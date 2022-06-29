#!/bin/bash
CONFIG="configs/lowres/lowres.gin"
EXP_NAME=$1
WORKDIR="/path/to/exp/$EXP_NAME"  # CHANGEME

python -m main \
  --gin_config="$CONFIG" \
  --mode="train" \
  --workdir="$WORKDIR" \
