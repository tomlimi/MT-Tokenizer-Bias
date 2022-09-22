import time
from transformers import MarianMTModel, MarianModel, MarianTokenizer
import torch
from typing import List
import logging

import json
import itertools
import copy


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

src_lang = "en"
tgt_lang = "de"
translator = "opus-mt"
average_embeddings=False

model_name = f"Helsinki-NLP/{translator}-{src_lang}-{tgt_lang}"

tokenizer_org = MarianTokenizer.from_pretrained(model_name)
model_org = MarianMTModel.from_pretrained(model_name)

tokenizer = copy.deepcopy(tokenizer_org)
model = copy.deepcopy(model_org)

variant_fn = f"../data/wino_mt/{tgt_lang}_variants.json"
with open(variant_fn, 'r') as var_json:
    variants = json.load(var_json)

variant_combined = list(itertools.chain.from_iterable(variants.values()))

add_tgt_words(variant_combined, tokenizer, model, average_embeddings)

if average_embeddings:
    model.save_pretrained(f"../models/model/{translator}-{src_lang}-{tgt_lang}-avg_emb")
else:
    model.save_pretrained(f"../models/model/{translator}-{src_lang}-{tgt_lang}-rand_emb")
tokenizer.save_pretrained(f"../models/tokenizer/with_professions_{translator}-{src_lang}-{tgt_lang}")