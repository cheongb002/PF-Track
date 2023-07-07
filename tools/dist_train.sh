#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28500}
JOB_STR="pf_track-$(date +%y-%m-%d_%H-%M-%S)"


TORCH_CUDNN_V8_API_ENABLED=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
--nproc_per_node=$GPUS \
--master_port=$PORT \
$(dirname "$0")/train.py \
$CONFIG \
--work-dir work_dirs/${JOB_STR} \
--launcher pytorch \
${@:3}
