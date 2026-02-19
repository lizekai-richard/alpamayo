#!/bin/bash
# Run dump_data.py for selected clip IDs.
# Clip IDs are read from clips_for_train.json in the repo root.

cd /home/zekail/alpamayo

torchrun --nproc_per_node=8 src/alpamayo_r1/dump_data.py \
  --clip_ids_file clips_for_train.json \
  --output_dir ./streaming_training_data