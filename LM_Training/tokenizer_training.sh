head -100l bibmed/title_abstract.txt > bibmed/title_abstract_100.txt
python3 -m venv my_test
source my_test/bin/activate
pip install sentencepiece

mkdir my_test_rep

TEXT_PATH=$HOME/bibmed/title_abstract_100.txt
TOKENIZER_PATH=$HOME/my_test_rep/clinical_pt

spm_train \
    --input=$TEXT_PATH \
     --model_prefix=$TOKENIZER_PATH \
     --vocab_size=320 \
     --character_coverage=0.9995 \
     --train_extremely_large_corpus=True

# you should have a model here
ls $HOME/my_test_rep/


# this is the code that use the original sentencepiece implementation

# install
source my_env/bin/activate
pip install sentencepiece

# try on a small proportion of the dataset
head -100l bibmed/title_abstract.txt > bibmed/title_abstract_100.txt

TEXT_PATH=$HOME/bibmed/title_abstract_10000.txt
TOKENIZER_PATH=$HOME/Language_model_training/English/Tokenizer/bio_english_sample

mkdir Language_model_training/English/Tokenizer
# train sentencepiece:
spm_train --input=$TEXT_PATH --model_prefix=$TOKENIZER_PATH --vocab_size=3200 --character_coverage=0.9995 --model_type=unigram --max_sentence_length=10000
# trainer_interface.cc(604) LOG(INFO) Saving model: /home/jknafou/Language_model_training/English/Tokenizer/bio_english_sample.model
# trainer_interface.cc(615) LOG(INFO) Saving vocabs: /home/jknafou/Language_model_training/English/Tokenizer/bio_english_sample.vocab

head -1l bibmed/title_abstract.txt | spm_encode --model=$TOKENIZER_PATH.model
head -1l bibmed/title_abstract.txt | spm_encode --model=$TOKENIZER_PATH.model --output_format=id
head -1l bibmed/title_abstract.txt | spm_encode --model=$TOKENIZER_PATH.model --output_format=id | spm_decode --model=$TOKENIZER_PATH.model --input_format=id

# complete training
# shuffle text file by line
# shuf -o $HOME/bibmed/title_abstract_shuffled.txt < $HOME/bibmed/title_abstract.txt
# head -10000000l bibmed/title_abstract_shuffled.txt > bibmed/title_abstract_10M.txt
# rm bibmed/title_abstract_shuffled.txt

source my_env/bin/activate
TEXT_PATH=$HOME/bibmed/title_abstract_10M.txt
TOKENIZER_PATH=$HOME/Language_model_training/English/Tokenizer/bio_english
LOG_PATH=$HOME/Language_model_training/English/Tokenizer/log_final.out
nohup spm_train \
    --input=$TEXT_PATH \
     --model_prefix=$TOKENIZER_PATH \
     --vocab_size=32000 \
     --character_coverage=0.9995 \
     --train_extremely_large_corpus=True &> $LOG_PATH &

tail -f $LOG_PATH
# Took a few hours to finish (~4h) on a machine with 600GB of RAM (doesn't work otherwise)
