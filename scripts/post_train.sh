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
DATA_NAME=$2

source /lnet/work/people/limisiewicz/mt-tokenizer-bias/.virtualenv/bin/activate
cd /lnet/work/people/limisiewicz/mt-tokenizer-bias/MT-Tokenizer-Bias || exit
# Meaning of acronyms:
# ft: fine-tuning (only embeddings layer)
# re: randomly intialized embeddings for added profession words
# ae: averaged embeddings for added profession words based on their constituting subword representations
# es: early stopping (default patience 5)
# wp: with professions only train on sentences containing profession name in source
# st: stronger training, learning rate 5e-5, epochs=3.0
# rae: reset all embeddings
# fe: freeze embeddings

OUTPUT_DIR="models/model/opus-mt-en-${TGT_LANG}-${DATA_NAME}-ft_ae_wp_st_fe"
mkdir ${OUTPUT_DIR}
#
python src/run_translation.py --model_name_or_path "models/model/opus-mt-en-${TGT_LANG}-avg_emb" \
 --tokenizer_name "models/tokenizer/with_professions_opus-mt-en-${TGT_LANG}" --do_train --do_eval --max_source_length 512 \
 --dataset_name ${DATA_NAME} --source_lang en --target_lang ${TGT_LANG} --dataset_config_name ${TGT_LANG}-en \
 --output_dir ${OUTPUT_DIR} --per_device_train_batch_size=16 \
 --per_device_eval_batch_size=16  --predict_with_generate \
 --save_total_limit 3 --save_steps 20000 --num_train_epochs=3.0 --report_to "tensorboard" --freeze --freeze_embeddings\
 --evaluation_strategy "steps" --eval_steps 20000 --learning_rate 5e-5 \
 --with_profession_only True --reset_all_embeddings False --preprocessing_num_workers=4
 # --early_stopping 5 --metric_for_best_model "loss" --greater_is_better False \
 #--reset_all_embeddings True --preprocessing_num_workers=4

#OUTPUT_DIR="models/model/opus-mt-en-${TGT_LANG}-${DATA_NAME}-ft_es_st_rea"
#mkdir ${OUTPUT_DIR}
#
#python src/run_translation.py --model_name_or_path "Helsinki-NLP/opus-mt-en-${TGT_LANG}" \
#  --do_train --do_eval --max_source_length 512 \
#  --dataset_name ${DATA_NAME} --source_lang en --target_lang ${TGT_LANG} --dataset_config_name ${TGT_LANG}-en \
#  --output_dir ${OUTPUT_DIR} --per_device_train_batch_size=16 \
#  --per_device_eval_batch_size=16  --predict_with_generate \
#  --save_total_limit 3 --save_steps 2000 --num_train_epochs=3.0 --report_to "tensorboard" --freeze \
#  --evaluation_strategy "steps" --eval_steps 2000 --learning_rate 5e-5 --early_stopping 5 --metric_for_best_model "loss" --greater_is_better False \
#   --with_profession_only False --reset_all_embeddings True --preprocessing_num_workers=4

