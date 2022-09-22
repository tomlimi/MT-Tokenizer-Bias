from tqdm import tqdm
from transformers import MarianMTModel, MarianModel, MarianTokenizer
import numpy as np
import torch


def translate_batch(batch_sentences, model, tokenizer, beam_size = 5, **kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    inputs = tokenizer(batch_sentences, truncation=True, padding=True, max_length=None, return_tensors="pt")
    # print(inputs)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=beam_size, **kwargs)
        output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return output


def translate_sentences(sentences, model, tokenizer,
                        show_progress_bar: bool = True, beam_size: int = 5, batch_size: int = 32, **kwargs):
    
    output = []
    
    #Sort by length to speed up processing
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    
    iterator = range(0, len(sentences_sorted), batch_size)
    if show_progress_bar:
        scale = min(batch_size, len(sentences))
        iterator = tqdm(iterator, total=len(sentences)/scale, unit_scale=scale, smoothing=0)
    
    for start_idx in iterator:
        output.extend(translate_batch(sentences_sorted[start_idx:start_idx+batch_size],model, tokenizer ,beam_size=beam_size, **kwargs))
    
    #Restore original sorting of sentences
    output = [output[idx] for idx in np.argsort(length_sorted_idx)]
    
    return output


src_lang = "en"
tgt_lang = "de"
translator = "opus-mt"

model = MarianMTModel.from_pretrained(f"../models/model/{translator}-{src_lang}-{tgt_lang}-ft_handcrafted_re/")
tokenizer = MarianTokenizer.from_pretrained(f"../models/tokenizer/with_professions_{translator}-{src_lang}-{tgt_lang}/")
#tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-de")

with open("../data/en.txt","r") as f:
    lines = f.readlines()
    lines = [l.split("\t")[2] for l in lines]
with open(f"../data/{translator}-ft_handcrafted_re/{src_lang}-{tgt_lang}.txt","w+") as f_out:
    translated= translate_sentences(lines, model, tokenizer)
    for orginal_sent, translated_sent in zip(lines, translated):
        f_out.write(orginal_sent+" ||| "+translated_sent+"\n")

