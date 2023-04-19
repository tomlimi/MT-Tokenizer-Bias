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
KEEP_PROFESSIONS=$3
#workdir=/lnet/work/people/limisiewicz/mt-tokenizer-bias
workdir=/cs/usr/matanel.oren/Desktop/bar
#source /lnet/work/people/limisiewicz/mt-tokenizer-bias/.virtualenv/bin/activate
source ${workdir}/MT-Tokenizer-Bias/venv/venv/bin/activate
cd ${workdir}/MT-Tokenizer-Bias || exit

if [ $KEEP_PROFESSIONS -eq 1 ]; then
  echo "Splitting professions; Target language: ${TGT_LANG} dataset: ${DATA_NAME}"
  OUTPUT_DIR="models/model/new-opus-mt-en-${TGT_LANG}-${DATA_NAME}-keep-professions"
  TOKENIZER="models/tokenizer/with_professions_opus-mt-en-${TGT_LANG}"
else
  echo "Not splitting professions; Target language: ${TGT_LANG} dataset: ${DATA_NAME}"
  OUTPUT_DIR="models/model/new-opus-mt-en-${TGT_LANG}-${DATA_NAME}-split-professions"
  TOKENIZER="Helsinki-NLP/opus-mt-en-${TGT_LANG}"
fi
MODEL="Helsinki-NLP/opus-mt-en-${TGT_LANG}"


echo "Model ${MODEL}; tokenizer ${TOKENIZER}; OUTPUT_DIR ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}

if [ "$TGT_LANG" \< "en" ]; then
  DATA_CONFIG_NAME="${TGT_LANG}-en"
else
  DATA_CONFIG_NAME="en-${TGT_LANG}"
fi

# max lr lowered for HE because of convergence issue
if [ "$TGT_LANG" = "he" ]; then
     LR=2e-4
else
     LR=3e-4
fi

python src/run_translation.py --model_name_or_path $MODEL --tokenizer_name $TOKENIZER \
 --do_train --do_eval --do_predict --train_from_scratch True --max_source_length 512 \
 --dataset_name ${DATA_NAME} --source_lang en --target_lang ${TGT_LANG} --dataset_config_name ${DATA_CONFIG_NAME} \
 --output_dir ${OUTPUT_DIR} --per_device_train_batch_size=16 \
 --per_device_eval_batch_size=16  --predict_with_generate \
 --save_total_limit 3 --save_steps 10000 --num_train_epochs=10.0 --max_grad_norm 5.0 --warmup_steps 16000 \
 --learning_rate ${LR} --label_smoothing_factor 0.1 --generation_num_beams 12 \
 --evaluation_strategy "steps" --eval_steps 10000 \
 --early_stopping 10 --metric_for_best_model "loss" --greater_is_better False --preprocessing_num_workers=4 \
 --load_best_model_at_end True --report_to "tensorboard"