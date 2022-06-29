import json
import os
from collections import defaultdict
import subprocess


def get_lexical_translations(directory, src_lang, tgt_lang, translator='gold'):
    """ Run fast align forward and save the alignment file. Saved aligned words to JSON"""

    file_prefix = directory + '/' + src_lang + '-' + tgt_lang + "_" + translator
    tok_file = file_prefix + ".tok"
    f_aligned_file = file_prefix + ".falign"
    r_aligned_file = file_prefix + ".ralign"
    aligned_file = file_prefix + ".align"
    json_file = file_prefix + "_lexical_translations.json"

    # Running fast_align
    # TODO: decide how to align reverse or two way

    # subprocess.call(f"fast_align -i {tok_file} -d -o -v > {f_aligned_file}", shell=True)
    # subprocess.call(f"fast_align -i {tok_file} -d -o -v -r > {r_aligned_file}", shell=True)
    # subprocess.call(f"atools -i {f_aligned_file} -j {r_aligned_file} -c grow-diag-final-and > {aligned_file}", shell=True)

    subprocess.call(f"fast_align -i {tok_file} -d -o -v -r > {aligned_file}", shell=True)

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
    with open(json_file , 'w', encoding='utf8') as json_s:
        json.dump(lexical_translations, json_s, indent=2, ensure_ascii=False)




