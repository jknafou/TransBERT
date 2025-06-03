#!/bin/bash

export GPU_DIR=./GPU
rm -rf $GPU_DIR && mkdir $GPU_DIR
TOTAL_GPU=$(lspci | grep -ci nvidia)
for ((i=0;i<TOTAL_GPU;i++)); do
  touch $GPU_DIR/$i
done
nbrun=16

models=`cat models.txt | grep -v "#" | tr "\n" " "`

for i in $(seq 1 $nbrun); do
  for model_name in $models; do
    while true; do
      sleep 5
      I_GPU="$(ls $GPU_DIR/ | head -n 1)"
      if [[ -n "$I_GPU" ]]; then
        rm $GPU_DIR/$I_GPU && CUDA_VISIBLE_DEVICES=$I_GPU ./run.sh $model_name &> $i\.out && touch $GPU_DIR/$I_GPU &
        break
      else
        sleep 10
      fi
    done
  done
done