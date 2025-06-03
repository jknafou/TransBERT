#!/usr/bin/env bash
# Apache 2.0

python task_2_finetuning_bert_cls_hpo_cv_v2.py --config="../yaml/cls_hpo.yaml" --model="$1" --fold="$2"
