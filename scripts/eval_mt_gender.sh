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
WINOMT_DIR=$4
FT=$5
# WINOMT_DIR="../../mt_gender"
if [ "$FT" == "1" ]; then
  FT_DATASET="variants-sents"
else
  FT_DATASET=""
  METHOD=""
fi


TEST_DATASET="opus100"
EXPERIMENT_NAME="${TRANSLATOR}-${FT_DATASET}-${METHOD}"
if [ "$FT" == "0" ]; then
  if [ "$TRANSLATOR" == "mbart50" ]; then
    MODEL="facebook/mbart-large-50-many-to-many-mmt"
  else
    MODEL="Helsinki-NLP/${TRANSLATOR}-en-${TGT_LANG}"
  fi
else
  MODEL="../models/model/${TRANSLATOR}-en-${TGT_LANG}-${FT_DATASET}-${METHOD}"
fi
OUTPUT_DIR="../results/${EXPERIMENT_NAME}"
echo "OUTPUT_DIR ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}

if [ "$TGT_LANG" \< "en" ]; then
  DATA_CONFIG_NAME="${TGT_LANG}-en"
else
  DATA_CONFIG_NAME="en-${TGT_LANG}"
fi

source ../venv/bin/activate

python run_translation.py --model_name_or_path $MODEL \
 --do_predict --max_source_length 512 \
 --dataset_name ${TEST_DATASET} --source_lang en --target_lang ${TGT_LANG} --dataset_config_name ${DATA_CONFIG_NAME} \
 --output_dir "${OUTPUT_DIR}/${TEST_DATASET}-en-${TGT_LANG}" --predict_with_generate

echo "python translate_custom.py --tgt_lang ${TGT_LANG} --translator ${TRANSLATOR} --method "${FT_DATASET}-${METHOD}""
python translate_custom.py --tgt_lang $TGT_LANG --translator $TRANSLATOR --method "${FT_DATASET}-${METHOD}"

echo "Entering WINOMT directory at ${WINOMT_DIR}."
mkdir -p  "${WINOMT_DIR}/translations/${EXPERIMENT_NAME}"
cp -r "${OUTPUT_DIR}/"* "${WINOMT_DIR}/translations/${EXPERIMENT_NAME}"

cd "${WINOMT_DIR}/src" || exit
#source ../venv/bin/activate
export FAST_ALIGN_BASE=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/fast_align
echo "Gender evaluation on all sentences"
../scripts/evaluate_language.sh ../data/aggregates/en.txt ${TGT_LANG} $EXPERIMENT_NAME

echo "Gender evaluation on pro-sterotypical sentences"
../scripts/evaluate_language.sh ../data/aggregates/en_pro.txt ${TGT_LANG} $EXPERIMENT_NAME

echo "Gender evaluation on anti-stereotypical sentences"
../scripts/evaluate_language.sh ../data/aggregates/en_anti.txt ${TGT_LANG} $EXPERIMENT_NAME

cd - || exit
cp -r "${WINOMT_DIR}/translations/${EXPERIMENT_NAME}/"* "${OUTPUT_DIR}"

