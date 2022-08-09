import json
import os
from collections import defaultdict
import subprocess


AWESOME_ALIGN_MODEL = "bert-base-multilingual-cased"


def get_lexical_translations(directory, src_lang, tgt_lang, translator='gold'):
    """ Run fast align forward and save the alignment file. Saved aligned words to JSON"""

    file_prefix = directory + '/' + src_lang + '-' + tgt_lang + "_" + translator
    tok_file = file_prefix + ".tok"
    aligned_file = file_prefix + ".align"
    json_file = file_prefix + "_lexical_translations.json"

    # Running awesome_align
    subprocess.run(f"CUDA_VISIBLE_DEVICES=0 awesome-align \
                    --output_file={aligned_file} \
                    --model_name_or_path={AWESOME_ALIGN_MODEL} \
                    --data_file={tok_file} \
                    --extraction 'softmax' \
                    --batch_size 32", shell=True)

    lexical_translations =[]
    with open(tok_file, 'r') as in_toks, open(aligned_file, 'r') as in_align:
        for src_tgt_toks, alignments in zip(in_toks, in_align):

            lt_line = defaultdict(list)

            src_toks, tgt_toks = src_tgt_toks.split(' ||| ')

            src_toks = src_toks.strip().split(' ')
            tgt_toks = tgt_toks.strip().split(' ')

            for alignment in alignments.split(' '):
                src_algn, tgt_algn = alignment.split('-')
                lt_line[src_toks[int(src_algn)]] = tgt_toks[int(tgt_algn)]

            lexical_translations.append(lt_line)

    lexical_translations = {lt_idx: lt_line for lt_idx, lt_line in enumerate(lexical_translations)}
    with open(json_file, 'w', encoding='utf8') as json_s:
        json.dump(lexical_translations, json_s, indent=2, ensure_ascii=False)




