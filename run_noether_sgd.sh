#!/bin/bash
set -e  # exit on error
for f in config_files/noether_runs_sgd/*.sh; do
  CUDA_VISIBLE_DEVICES=8 bash "$f"
done