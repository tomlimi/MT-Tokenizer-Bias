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
TRANSLATOR=$2
DATASET=$3
METHOD=$4

TEST_DATASET="opus100"
EXPERIMENT_NAME="${TRANSLATOR}-${DATASET}-${METHOD}"
MODEL="../models/model/${TRANSLATOR}-en-${TGT_LANG}-${DATASET}-${METHOD}"
OUTPUT_DIR="../data/${EXPERIMENT_NAME}/${TEST_DATASET}-en-${TGT_LANG}"

source /lnet/work/people/limisiewicz/mt-tokenizer-bias/.virtualenv/bin/activate
cd /lnet/work/people/limisiewicz/mt-tokenizer-bias/MT-Tokenizer-Bias/src || exit

mkdir -p ${OUTPUT_DIR}

if [ "$TGT_LANG" \< "en" ]; then
  DATA_CONFIG_NAME="${TGT_LANG}-en"
else
  DATA_CONFIG_NAME="en-${TGT_LANG}"
fi

python run_translation.py --model_name_or_path $MODEL \
 --do_predict --max_source_length 512 \
 --dataset_name ${TEST_DATASET} --source_lang en --target_lang ${TGT_LANG} --dataset_config_name ${DATA_CONFIG_NAME} \
 --output_dir ${OUTPUT_DIR} --predict_with_generate

python translate_custom.py --tgt_lang $TGT_LANG --translator $TRANSLATOR --method "${DATASET}-${METHOD}"

source /lnet/work/people/limisiewicz/mt_gender_bar/.virturalenvs/bin/activate
cd /lnet/work/people/limisiewicz/mt-tokenizer-bias/mt_gender/src || exit
mkdir "../translations/${EXPERIMENT_NAME}"
cp -r "/lnet/work/people/limisiewicz/mt-tokenizer-bias/MT-Tokenizer-Bias/data/${EXPERIMENT_NAME}/"* "../translations/${EXPERIMENT_NAME}"

../scripts/evaluate_all_languages.sh ../data/aggregates/en.txt $EXPERIMENT_NAME


#python run_translation.py --model_name_or_path "Helsinki-NLP/opus-mt-en-de" \
# --do_predict --max_source_length 512 \
# --dataset_name "opus100" --source_lang en --target_lang "de" --dataset_config_name "de-en" \
# --output_dir "../data/opus-mt/opus100-en-de" --predict_with_generate
#
# python run_translation.py --model_name_or_path "Helsinki-NLP/opus-mt-en-he" \
#  --do_predict --max_source_length 512 \
#  --dataset_name "opus100" --source_lang "en" --target_lang "he" --dataset_config_name "en-he" \
#  --output_dir "../data/opus-mt/opus100-en-he" --predict_with_generate

