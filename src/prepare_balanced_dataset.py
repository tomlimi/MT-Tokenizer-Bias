import json
import os
from argparse import ArgumentParser

TEMPLATES = {"de": {"male": "Er ist der {} .",
                    "female": "Sie ist die {} ."
                    },
             "he": {"male": "הוא {} .",
                    "female": "היא {} ."
                    },
             "en": {"male": "He is the {} .",
                    "female": "She is the {} ."
                    },
             
             "es": {"male": "Él es el {} .",
                    "female": "Ella es la {} ."
                    }
             }

if __name__ == "__main__":
    
    args = ArgumentParser()
    
    args.add_argument("--src_lang", type=str, default="en", required=False)
    args.add_argument("--tgt_lang", type=str, required=True)
    args.add_argument("--variants_dir", type=str, default="../data/wino_mt/",
                      help="Directory where the profession variants are stored")
    args.add_argument("--out_dir", type=str, default="../data/gender-debias-variants-dataset/")
    
    args = args.parse_args()
    
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    
    variant_fn = os.path.join(args.variants_dir, f"{tgt_lang}_variants.json")

    with open(variant_fn, 'r') as var_json:
        variants = json.load(var_json)

    words_fn = os.path.join(args.out_dir, f"words.en{tgt_lang}.json")
    os.makedirs(os.path.dirname(words_fn), exist_ok=True)
    sentence_fn = os.path.join(args.out_dir, f"sentences.en{tgt_lang}.json")
    os.makedirs(os.path.dirname(sentence_fn), exist_ok=True)
    with open(words_fn, 'w') as w_out, open(sentence_fn, 'w') as s_out:
        for profession_gender, translations in variants.items():
            src_prof, gender = profession_gender.split('-')
            for tgt_prof in translations:
                word_ex = {"translation": {src_lang: src_prof, tgt_lang: tgt_prof}}
                sent_ex = {"translation": {src_lang: TEMPLATES[src_lang][gender].format(src_prof),
                                           tgt_lang: TEMPLATES[tgt_lang][gender].format(tgt_prof)}}
                w_out.write(json.dumps(word_ex, ensure_ascii=False) + '\n')
                s_out.write(json.dumps(sent_ex, ensure_ascii=False) + '\n')
