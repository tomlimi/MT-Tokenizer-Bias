#!/bin/bash

TGT_LANG=$1
KEEP_PROFESSIONS=$2
TRANSLATOR=$3
DATA_DIR="../data"
MODEL_DIR="../models/model"
TOKENIZER_DIR="../models/tokenizer"

# WORDFILE_NAME="${DATA_DIR}/gender-debias-variants-dataset/words.en${TGT_LANG}.json"
SENTFILE_NAME="${DATA_DIR}/gender-debias-variants-dataset/sentences.en${TGT_LANG}.json"

if [ $KEEP_PROFESSIONS -eq 1 ]; then
  echo "Splitting professions; Target language: ${TGT_LANG} dataset: ${WORDFILE_NAME}"
  # OUTPUT_DIR_WORDS="${MODEL_DIR}/${TRANSLATOR}-en-${TGT_LANG}-variants-words-re"
  OUTPUT_DIR_SENTS="${MODEL_DIR}/${TRANSLATOR}-en-${TGT_LANG}-variants-sents-re"
  MODEL="${MODEL_DIR}/${TRANSLATOR}-en-${TGT_LANG}-rand_emb"
  TOKENIZER="${TOKENIZER_DIR}/with_professions_${TRANSLATOR}-en-${TGT_LANG}"
else
  echo "Don't splitting professions; Target language: ${TGT_LANG} dataset: ${WORDFILE_NAME}"
  # OUTPUT_DIR_WORDS="${MODEL_DIR}/${TRANSLATOR}-en-${TGT_LANG}-variants-words-"
  OUTPUT_DIR_SENTS="${MODEL_DIR}/${TRANSLATOR}-en-${TGT_LANG}-variants-sents-"
  if [ $TRANSLATOR == "opus-mt" ]; then
    MODEL="Helsinki-NLP/${TRANSLATOR}-en-${TGT_LANG}"
    TOKENIZER="Helsinki-NLP/${TRANSLATOR}-en-${TGT_LANG}"
  else
    MODEL="facebook/mbart-large-50-many-to-many-mmt"
    TOKENIZER="facebook/mbart-large-50-many-to-many-mmt"
  fi
fi

echo "Fine-tuning model ${MODEL} with tokenizer ${TOKENIZER} on dataset ${SENTFILE_NAME}"

mkdir -p ${OUTPUT_DIR_SENTS}

python run_translation.py --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER --do_train --max_source_length 512 \
  --source_lang en --target_lang ${TGT_LANG} --train_file $SENTFILE_NAME \
  --output_dir ${OUTPUT_DIR_SENTS} --per_device_train_batch_size=1 \
  --predict_with_generate \
  --save_total_limit 3 --save_steps 2000 --num_train_epochs=3.0 --report_to "tensorboard" --freeze \
  --learning_rate 5e-5  --lr_scheduler_type linear \
  --with_profession_only False  --preprocessing_num_workers=4