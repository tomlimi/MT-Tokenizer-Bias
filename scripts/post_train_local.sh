#!/bin/bash
CUDA_version=11.3
CUDNN_version=8.2
CUDA_DIR_OPT=/opt/cuda/$CUDA_version
if [ -d "$CUDA_DIR_OPT" ] ; then
  CUDA_DIR=$CUDA_DIR_OPT
  export CUDA_HOME=$CUDA_DIR
  export THEANO_FLAGS="cuda.root=$CUDA_HOME,device=gpu,floatX=float32"
  export PATH=$PATH:$CUDA_DIR/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIR/cudnn/$CUDNN_version/lib64:$CUDA_DIR/lib64
  export CPATH=$CUDA_DIR/cudnn/$CUDNN_version/include:$CPATH
fi

TGT_LANG=$1
KEEP_PROFESSIONS=$2
WORDFILE_NAME="data/gender-debias-variants-dataset/words.en${TGT_LANG}.json"
SENTFILE_NAME="data/gender-debias-variants-dataset/sentences.en${TGT_LANG}.json"

source /lnet/work/people/limisiewicz/mt-tokenizer-bias/.virtualenv/bin/activate
cd /lnet/work/people/limisiewicz/mt-tokenizer-bias/MT-Tokenizer-Bias || exit


if [ $KEEP_PROFESSIONS -eq 1 ]; then
  echo "Splitting professions; Target language: ${TGT_LANG} dataset: ${WORDFILE_NAME}"
  OUTPUT_DIR_WORDS="models/model/opus-mt-en-${TGT_LANG}-variants-words-re"
  OUTPUT_DIR_SENTS="models/model/opus-mt-en-${TGT_LANG}-variants-sents-re"
  MODEL="models/model/opus-mt-en-${TGT_LANG}-rand_emb"
  TOKENIZER="models/tokenizer/with_professions_opus-mt-en-${TGT_LANG}"
elif [ $KEEP_PROFESSIONS -eq 2 ]; then
    echo "Splitting professions; Target language: ${TGT_LANG} dataset: ${WORDFILE_NAME}"
    OUTPUT_DIR_WORDS="models/model/opus-mt-en-${TGT_LANG}-variants-words-ae"
    OUTPUT_DIR_SENTS="models/model/opus-mt-en-${TGT_LANG}-variants-sents-ae"
    MODEL="models/model/opus-mt-en-${TGT_LANG}-avg_emb"
    TOKENIZER="models/tokenizer/with_professions_opus-mt-en-${TGT_LANG}"
else
  echo "Don't splitting professions; Target language: ${TGT_LANG} dataset: ${WORDFILE_NAME}"
  OUTPUT_DIR_WORDS="models/model/opus-mt-en-${TGT_LANG}-variants-words-"
  OUTPUT_DIR_SENTS="models/model/opus-mt-en-${TGT_LANG}-variants-sents-"
  MODEL="Helsinki-NLP/opus-mt-en-${TGT_LANG}"
  TOKENIZER="Helsinki-NLP/opus-mt-en-${TGT_LANG}"
fi

#OUTPUT_DIR_WORDS="${OUTPUT_DIR_WORDS}-const_lr_"
OUTPUT_DIR_SENTS="${OUTPUT_DIR_SENTS}-freeze_embeddings"
# mkdir ${OUTPUT_DIR_WORDS}
#
#python src/run_translation.py --model_name_or_path  $MODEL \
# --tokenizer_name $TOKENIZER --do_train  --max_source_length 512 \
# --source_lang en --target_lang ${TGT_LANG} --train_file $WORDFILE_NAME \
# --output_dir ${OUTPUT_DIR_WORDS} --per_device_train_batch_size=1 \
# --predict_with_generate \
# --save_total_limit 3 --save_steps 2000 --num_train_epochs=3.0 --report_to "tensorboard" --freeze \
# --learning_rate 5e-5 --lr_scheduler_type constant \
# --with_profession_only False --preprocessing_num_workers=4



mkdir ${OUTPUT_DIR_SENTS}

python src/run_translation.py --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER --do_train --max_source_length 512 \
  --source_lang en --target_lang ${TGT_LANG} --train_file $SENTFILE_NAME \
  --output_dir ${OUTPUT_DIR_SENTS} --per_device_train_batch_size=8 \
  --predict_with_generate \
  --save_total_limit 3 --save_steps 2000 --num_train_epochs=10.0 --report_to "tensorboard" --freeze --freeze_embeddings\
  --learning_rate 5e-5 --lr_scheduler_type linear \
  --with_profession_only False  --preprocessing_num_workers=4