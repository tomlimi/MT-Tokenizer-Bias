from transformers import pipeline
from mosestokenizer import MosesTokenizer
import logging
import argparse

from word_alignment_utils import get_lexical_translations


def translate_file(dir, file, src_lang, tgt_lang, src_first, translator):
    """ Translate sentences and save in a format ready to be processed by fast_align """
    src_lines = []

    in_file = dir + '/' + file
    out_file = dir + '/' + src_lang + '-' + tgt_lang + "_" + translator + ".tok"

    translation_pipeline = pipeline("translation", model=translator)

    with open(in_file, 'r') as in_s:
        for line in in_s:
            line = line.strip()
            split_line = line.split("\t")
            if src_first:
                src_lines.append(split_line[2])
            else:
                src_lines.append(split_line[3])

    logging.info("Started translating!")
    output = translation_pipeline(src_lines)
    out_lines = [out_line['translation_text'] for out_line in output]
    logging.info("Translation done!")

    src_tokenizer = MosesTokenizer(src_lang)
    out_tokenizer = MosesTokenizer(tgt_lang)

    with open(out_file, 'w') as out_s:
        for src_line, out_line in zip(src_lines, out_lines):
            src_tokenized = ' '.join(src_tokenizer(src_line))
            out_tokenized = ' '.join(out_tokenizer(out_line))

            out_s.write(src_tokenized + ' ||| ' + out_tokenized +'\n')

    src_tokenizer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/tatoeba', required=False)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--src_lang', type=str, default='en', required=False)
    parser.add_argument('--tgt_lang', type=str, required=True)
    parser.add_argument('--src_first', type=bool)
    parser.add_argument('--translator', type=str, required=True)
    args = parser.parse_args()

    translate_file(args.dir, args.file, args.src_lang, args.tgt_lang, args.src_first, args.translator)
    get_lexical_translations(args.dir, args.src_lang, args.tgt_lang, translator=args.translator)
