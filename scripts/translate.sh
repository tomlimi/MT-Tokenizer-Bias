#!/bin/bash
CUDA_version=10.1
CUDNN_version=7.6
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

source /lnet/work/people/limisiewicz/mt-tokenizer-bias/.virturalenvs/bin/activate
cd /lnet/work/people/limisiewicz/mt-tokenizer-bias/MT-Tokenizer-Bias || exit

if [ "$TGT_LANG" == "he" ]; then
  python src/translate_word_align.py --file tatoeba-test-v2021-08-07.eng-heb.txt --tgt_lang $TGT_LANG --src_first True --translator opus-mt
elif [ "$TGT_LANG" == "de" ]; then
  python src/translate_word_align.py --file tatoeba-test-v2021-08-07.deu-eng.txt --tgt_lang $TGT_LANG  --translator opus-mt
else
  echo "Unsupported target language"
fi