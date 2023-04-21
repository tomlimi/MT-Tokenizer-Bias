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
METHOD=$3
FT_DATASET="variants-sents"
WINOMT_DIR="../../mt_gender"

TEST_DATASET="opus100"
EXPERIMENT_NAME="${TRANSLATOR}-${FT_DATASET}-${METHOD}"
MODEL="../models/model/${TRANSLATOR}-en-${TGT_LANG}-${FT_DATASET}-${METHOD}"
OUTPUT_DIR="../results/${EXPERIMENT_NAME}"

mkdir -p ${OUTPUT_DIR}

if [ "$TGT_LANG" \< "en" ]; then
  DATA_CONFIG_NAME="${TGT_LANG}-en"
else
  DATA_CONFIG_NAME="en-${TGT_LANG}"
fi

source ../.virtualenv/bin/activate

python run_translation.py --model_name_or_path $MODEL \
 --do_predict --max_source_length 512 \
 --dataset_name ${TEST_DATASET} --source_lang en --target_lang ${TGT_LANG} --dataset_config_name ${DATA_CONFIG_NAME} \
 --output_dir "${OUTPUT_DIR}/${TEST_DATASET}-en-${TGT_LANG}" --predict_with_generate

python translate_custom.py --tgt_lang $TGT_LANG --translator $TRANSLATOR --method "${FT_DATASET}-${METHOD}"

echo "Entering WINOMT directory at ${WINOMT_DIR}."
mkdir -p  "${WINOMT_DIR}/translations/${EXPERIMENT_NAME}"
cp -r "${OUTPUT_DIR}/"* "${WINOMT_DIR}/translations/${EXPERIMENT_NAME}"

cd "${WINOMT_DIR}/src" || exit
source ../.virtualenv/bin/activate

echo "Gender evaluation on all sentences"
../scripts/evaluate_language.sh ../data/aggregates/en.txt ${TGT_LANG} $EXPERIMENT_NAME

echo "Gender evaluation on pro-sterotypical sentences"
../scripts/evaluate_language.sh ../data/aggregates/en_pro.txt ${TGT_LANG} $EXPERIMENT_NAME

echo "Gender evaluation on anti-stereotypical sentences"
../scripts/evaluate_language.sh ../data/aggregates/en_anti.txt ${TGT_LANG} $EXPERIMENT_NAME

cd - || exit
cp -r "${WINOMT_DIR}/translations/${EXPERIMENT_NAME}/"* "${OUTPUT_DIR}"

