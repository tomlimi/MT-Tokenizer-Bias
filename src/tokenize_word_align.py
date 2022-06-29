from mosestokenizer import MosesTokenizer
import argparse

import os

from word_alignment_utils import get_lexical_translations

def tokenize_file(dir, file, src_lang, tgt_lang, src_first):
    """ Tokeznize languages and save in a format ready to be processed by fast_align """
    src_lines = []
    tgt_lines = []
    in_file = dir + '/' + file
    out_file = dir + '/' + src_lang + '-' + tgt_lang + "_gold.tok"

    with open(in_file, 'r') as in_s:
        for line in in_s:
            line = line.strip()
            split_line = line.split("\t")
            if src_first:
                src_lines.append(split_line[2])
                tgt_lines.append(split_line[3])
            else:
                tgt_lines.append(split_line[2])
                src_lines.append(split_line[3])

    src_tokenizer = MosesTokenizer(src_lang)
    tgt_tokenizer = MosesTokenizer(tgt_lang)

    with open(out_file, 'w') as out_s:
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_tokenized = ' '.join(src_tokenizer(src_line))
            tgt_tokenized = ' '.join(tgt_tokenizer(tgt_line))

            out_s.write(src_tokenized + ' ||| ' + tgt_tokenized +'\n')

    src_tokenizer.close()
    tgt_tokenizer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/tatoeba', required=False)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--src_lang', type=str, default='en', required=False)
    parser.add_argument('--tgt_lang', type=str, required=True)
    parser.add_argument('--src_first', type=bool)
    args = parser.parse_args()

    tokenize_file(args.dir, args.file, args.src_lang, args.tgt_lang, args.src_first)
    get_lexical_translations(args.dir, args.src_lang, args.tgt_lang)