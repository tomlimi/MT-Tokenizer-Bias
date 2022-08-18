import argparse
from collections import defaultdict
import json
from transformers import MarianTokenizer
from lexical_dictionary import LexicalDictionary


def load_lexical_translations(dir, src_lang, tgt_lang, translation_type):
    json_fn = dir + '/' + src_lang + '-' + tgt_lang + "_" + translation_type + "_lexical_translations.json"
    with open(json_fn, 'r') as json_file:
        translations = json.load(json_file)
        json_file.close()
        
    return translations


def evaluate_lexical_translation(dir, src_lang, tgt_lang, translator, lexical_dict, results):
    
    mt_trans = load_lexical_translations(dir, src_lang, tgt_lang, translator)
    gold_trans = load_lexical_translations(dir, src_lang, tgt_lang, "gold")
    
    assert len(mt_trans) == len(gold_trans)
    
    for sent_idx in range(len(mt_trans)):
        for (mt_src, mt_tgt), (gold_src, gold_tgt) in zip(mt_trans[str(sent_idx)], gold_trans[str(sent_idx)]):
            assert mt_src == gold_src
            if gold_tgt:
                if gold_tgt.lower() in lexical_dict[gold_src.lower()]:
                    results[gold_tgt]['count'] += 1
                    if gold_tgt == mt_tgt:
                        results[gold_tgt]['correct'] += 1
    
    return results


def count_tokens(src_lang, tgt_lang, translator, results):
    tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/{translator}-{src_lang}-{tgt_lang}")
    for tgt_word in results.keys():
        with tokenizer.as_target_tokenizer():
            tokenized = tokenizer(tgt_word)
            # TODO tokenizer always adds </s> token. Check if it's solved everywhere.
            num_tokens = len(tokenized['input_ids']) - 1
        results[tgt_word]['tokens'] = num_tokens
    
    return results
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/tatoeba', required=False)
    parser.add_argument('--translator', type=str)
    parser.add_argument('--src_lang', type=str, default='en', required=False)
    parser.add_argument('--tgt_lang', type=str, required=True)
    parser.add_argument("--lex_dict", type=str)
    args = parser.parse_args()
    
    lexical_dict = LexicalDictionary(args.lex_dict)
    results = defaultdict(lambda: defaultdict(int))
    
    results = evaluate_lexical_translation(args.dir,args.src_lang, args.tgt_lang, args.translator, lexical_dict, results)
    results = count_tokens(args.src_lang, args.tgt_lang, args.translator, results)
    
    out_file = args.dir + '/' + args.src_lang + '-' + args.tgt_lang + "_results.json"
    with open(out_file, 'w', encoding='utf8') as json_s:
        json.dump(results, json_s, indent=2, ensure_ascii=False)