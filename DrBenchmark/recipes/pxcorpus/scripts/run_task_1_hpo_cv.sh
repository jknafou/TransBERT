#!/usr/bin/env bash
# Apache 2.0

python task_1_finetuning_bert_ner_hpo_cv_v2.py --config="../yaml/ner_hpo.yaml" --model="$1" --fold="$2"
