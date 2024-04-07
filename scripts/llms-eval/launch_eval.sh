#!/bin/bash -l
conda activate llava
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --num_processes=2 -m lmms_eval --config=./scripts/llms-eval/eval_config_vqa.yaml 