"""
This script adds the profession variants to the tokenizer and embedding layer of the translation model.
"""

import time
from transformers import MarianMTModel, MarianModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch
from typing import List
import logging
from argparse import ArgumentParser
import os

import json
import itertools
import copy


mbert50_lang_map = {'en': 'en_XX', 'de': 'de_DE', 'he': 'he_IL', 'es': 'es_XX'}

def add_tgt_words(words, tokenizer, model,average_embeddings):
    
    tok_ids_per_word = {}
    with tokenizer.as_target_tokenizer():
        for word in words:
            token_ids = tokenizer(word)["input_ids"][:-1]
            if len(token_ids) == 1:
                print(f"Word: {word} already in the vocabulary")
            else:
                tok_ids_per_word[word] = token_ids

    num_added = tokenizer.add_tokens(list(tok_ids_per_word.keys()))
    sorted_words_added = list(sorted(tokenizer.added_tokens_encoder, key=tokenizer.added_tokens_encoder.get))
    assert num_added == len(sorted_words_added)
    
    model.resize_token_embeddings(len(tokenizer))
    
    if average_embeddings:
        lm_head = model.get_output_embeddings()
    
        with torch.no_grad():
            avg_embs = []
            for word in sorted_words_added:
                tok_ids = tok_ids_per_word[word]
                tok_weights = lm_head.weight[tok_ids,:]
            
                weight_mean = torch.mean(tok_weights, axis=0, keepdim=True)
                avg_embs.append(weight_mean)
        
            lm_head.weight[-num_added:,:] = torch.vstack(avg_embs).requires_grad_()
    
        model.set_output_embeddings(lm_head)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)


if __name__ == "__main__":

    args = ArgumentParser()
    
    args.add_argument("--src_lang", type=str, default="en")
    args.add_argument("--tgt_lang", type=str, default="de")
    args.add_argument("--translator", type=str, default="opus-mt",
                      help="Translation model that should be updated, should be an instacne of MarianMTModel")
    args.add_argument("--variants_dir", type=str, default="../data/wino_mt/",
                      help="Directory where the profession variants are stored")
    args.add_argument("--tokenizer_dir", type=str, default="../models/tokenizer",
                      help="Directory where the updated tokenizer is stored")
    args.add_argument("--model_dir", type=str, default="../models/model",
                      help="Directory where the updated model is stored")
    args.add_argument("--average_embeddings", type=bool, default=False)
    
    args = args.parse_args()
    
    translator = args.translator
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    average_embeddings= args.average_embeddings
    
    if args.translator == "opus-mt":
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer_org = MarianTokenizer.from_pretrained(model_name)
        model_org = MarianMTModel.from_pretrained(model_name)
    elif args.translator == "mbart50":
        model_name = f"facebook/mbart-large-50-many-to-many-mmt"
        tokenizer_org = MBart50Tokenizer.from_pretrained(model_name, src_lang=mbert50_lang_map[src_lang],
                                                         tgt_lang=mbert50_lang_map[tgt_lang])
        
        model_org = MBartForConditionalGeneration.from_pretrained(model_name)
    else:
        model_name = args.translator
        
    model_name += f"-{src_lang}-{tgt_lang}"
    
    tokenizer = copy.deepcopy(tokenizer_org)
    model = copy.deepcopy(model_org)
    
    variant_fn = os.path.join(args.variants_dir, f"{tgt_lang}_variants.json")
    if not os.path.exists(variant_fn):
        raise FileNotFoundError(f"Variants file {variant_fn} not found. Downoload provided profession variants list"
                                "By default they should be stored in ../data/wino_mt/")
    with open(variant_fn, 'r') as var_json:
        variants = json.load(var_json)
    
    variant_combined = list(itertools.chain.from_iterable(variants.values()))
    
    add_tgt_words(variant_combined, tokenizer, model, average_embeddings)
    
    if average_embeddings:
        model.save_pretrained(os.path.join(args.model_dir,f"{translator}-{src_lang}-{tgt_lang}-avg_emb"))
    else:
        model.save_pretrained(os.path.join(args.model_dir,f"{translator}-{src_lang}-{tgt_lang}-rand_emb"))
    tokenizer.save_pretrained(os.path.join(args.tokenizer_dir, f"with_professions_{translator}-{src_lang}-{tgt_lang}"))