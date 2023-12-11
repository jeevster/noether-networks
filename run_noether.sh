#!/bin/bash
set -e  # exit on error
for f in config_files/pinns/*.sh; do
  CUDA_VISIBLE_DEVICES=7 bash "$f"
done