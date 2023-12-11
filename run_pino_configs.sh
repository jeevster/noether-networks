#!/bin/bash
CUDA_VISIBLE_DEVICES=8 bash config_files/pino_runs/run_pino_original.sh;
CUDA_VISIBLE_DEVICES=8 bash config_files/pino_runs/run_pino_mse.sh;
CUDA_VISIBLE_DEVICES=8 bash config_files/pino_runs/run_pino_log.sh;
CUDA_VISIBLE_DEVICES=8 bash config_files/pino_runs/run_pino_mse_operator.sh;
CUDA_VISIBLE_DEVICES=8 bash config_files/pino_runs/run_pino_log_operator.sh;