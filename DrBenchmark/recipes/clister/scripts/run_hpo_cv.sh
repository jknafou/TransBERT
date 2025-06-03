#!/usr/bin/env bash
# Apache 2.0

python finetuning_bert_regr_hpo_cv_v2.py --config="../yaml/regr_hpo.yaml" --model="$1" --fold="$2"
