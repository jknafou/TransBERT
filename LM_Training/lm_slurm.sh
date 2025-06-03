#!/bin/bash
#SBATCH --job-name LM_TRAINING            # this is a parameter to help you sort your job when listing it
#SBATCH --error LM_TRAINING-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output LM_TRAINING-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00

cd SLURM
rm -r env

# for baobab
ml GCCcore/11.3.0 Python/3.10.4
ml CUDA/11.7.0
python3 -m venv env
source env/bin/activate
env/bin/python3 -m pip install --upgrade pip
pip3 install -I torch==2.0.1
pip3 install -I torchaudio==2.0.2
pip3 install -I torchvision==0.15.2
pip3 install -I sentencepiece
pip3 install -I tensorboard==2.12.1
pip3 install transformers==4.25.1

python --version
nvcc --version

echo number of process
nproc
nvidia-smi

export language=$1
export model_name=$2
export tokenizer=$3 
export PROJECT_DIR=.
export WD=../TransBioBERT
export DATA_DIR=$WD/data/$language
export data_file_path=$DATA_DIR/title_abstract.txt

eval_size=100000
max_steps=500000
eval_steps=10000
logging_steps=100
save_steps=10000
max_position_embeddings=512
n_process=3
n_subprocess=2

if [[ "$model_name" == "base" ]]; then
  learning_rate=6e-4
  per_device_train_batch_size=96
  gradient_accumulation_steps=28
  warmup_steps=24000
  hidden_size=768
  num_attention_heads=12
  num_hidden_layers=12
fi
if [[ "$model_name" == "large" ]]; then
  learning_rate=4e-4
  per_device_train_batch_size=42
  gradient_accumulation_steps=63
  warmup_steps=30000
  hidden_size=1024
  num_attention_heads=16
  num_hidden_layers=24
fi
if [[ "$tokenizer" != "pretrained" ]]; then
  model_name=$model_name\-tokenizer=$tokenizer
fi

rm $data_file_path\_train\_$model_name\0*
shuf -o $data_file_path\_train\_shuffled < $data_file_path\_train
mv $data_file_path\_train\_shuffled $data_file_path\_train


for split in train; do
    export size=$(stat -c%s $data_file_path\_$split)
    export process_size=$((size/n_process))
    split -C $process_size --numeric-suffixes -a 3 $data_file_path\_$split $data_file_path\_$split\_$model_name
    for path in $data_file_path\_$split\_$model_name\0*; do
        if [[ $path == *$n_process ]]; then
            rm $path
            continue
        fi
        export size=$(stat -c%s $path)
        export process_size=$((size/n_subprocess))
            process_size=$((process_size-1))
        split -C $process_size --numeric-suffixes $path $path
        rm $path
    done
    for path in $data_file_path\_$split\_$model_name*; do
        if [[ $path == *$n_subprocess ]]; then
            rm $path
            continue
        fi
    done
done

nvidia-smi -l 10 > nvidia-smi_$model_name.out &

echo starting training of $model_name
SCRIPT_PATH=$PROJECT_DIR/lm_training_gpu.py

nohup python -m torch.distributed.run --nproc_per_node $n_process $SCRIPT_PATH \
 $max_steps $eval_steps $logging_steps $save_steps $warmup_steps \
 $per_device_train_batch_size $gradient_accumulation_steps $learning_rate $language \
  $model_name $hidden_size $num_attention_heads $num_hidden_layers \
   $max_position_embeddings $WD $DATA_DIR/../.. 1 $tokenizer &> training_$model_name.out