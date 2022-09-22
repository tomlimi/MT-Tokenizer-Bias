import json
import argparse
import os


def convert_to_jsonl(in_file, out_jsonl, src_lang, tgt_lang):

    with open(in_file, 'r') as in_s, open(out_jsonl, 'w', encoding='utf8') as out_s:
        for line in in_s:
            line = line.strip()
            src_sent, tgt_sent = line.split("|")
            trans_dict = {"translation": {src_lang: src_sent, tgt_lang: tgt_sent}}
            json.dump(trans_dict, out_s, ensure_ascii=False)
            out_s.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/gender-debias-saunders', required=False)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--src_lang', type=str, default='en', required=False)
    parser.add_argument('--tgt_lang', type=str, required=True)
    args = parser.parse_args()
    
    in_file = os.path.join(args.dir, args.file)
    out_file = in_file +".jsonl"
    convert_to_jsonl(in_file, out_file, args.src_lang, args.tgt_lang)