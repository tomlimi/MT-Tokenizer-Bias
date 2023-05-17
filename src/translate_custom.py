from tqdm import tqdm
from transformers import MarianMTModel, MarianModel, MarianTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
import numpy as np
import torch
import os
import argparse
LANGUAGE_CODES_MAP = {"en":"en_XX", "es":"es_XX","he":"he_IL","de":"de_DE"}
def translate_batch(batch_sentences, model, tokenizer, beam_size = 5, **kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    inputs = tokenizer(batch_sentences, truncation=True, padding=True, max_length=None, return_tensors="pt")
    # print(inputs)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    with torch.no_grad():
        # generated_tokens = model.generate(**encoded_en,
        #                                   forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODES_MAP["he"]])
        translated = model.generate(**inputs, num_beams=beam_size,forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang], **kwargs)
        output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        n_sents_with_added_tokens = sum([any(t.cpu().data.numpy() >= tokenizer.vocab_size) for t in translated])
    return output, n_sents_with_added_tokens


def translate_sentences(sentences, model, tokenizer,
                        show_progress_bar: bool = True, beam_size: int = 5, batch_size: int = 32, **kwargs):
    
    n_sents_with_added_tokens = 0
    output = []
    
    # Sort by length to speed up processing
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    
    iterator = range(0, len(sentences_sorted), batch_size)
    if show_progress_bar:
        scale = min(batch_size, len(sentences))
        iterator = tqdm(iterator, total=len(sentences)/scale, unit_scale=scale, smoothing=0)
    
    for start_idx in iterator:
        translated, n_sents_with_added_tokens_per_batch = translate_batch(sentences_sorted[start_idx:start_idx+batch_size],model, tokenizer ,beam_size=beam_size, **kwargs)
        output.extend(translated)
        n_sents_with_added_tokens += n_sents_with_added_tokens_per_batch
        
    #Restore original sorting of sentences
    output = [output[idx] for idx in np.argsort(length_sorted_idx)]
    
    print(f"Number of sentences with added tokens: {n_sents_with_added_tokens}")
    
    return output, n_sents_with_added_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', type=str, default='en', required=False)
    parser.add_argument('--tgt_lang', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--translator', type=str, required=True)
    
    parser.add_argument('--model_dir', type=str, default='../models/model/', required=False)
    parser.add_argument('--tokenizer_dir', type=str, default='../models/tokenizer/', required=False)
    parser.add_argument('--output_dir', type=str, default='../results/', required=False)
    
    args = parser.parse_args()
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    method = args.method
    translator = args.translator
    
    model_dir = args.model_dir
    tokenizer_dir = args.tokenizer_dir
    output_dir = args.output_dir

    if method=="-":
        if translator == 'opus-mt':
            model = MarianMTModel.from_pretrained(f"Helsinki-NLP/{translator}-en-{tgt_lang}")

        elif translator == 'mbart50':
            model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        else:
            raise ValueError
    else:
        if translator == 'opus-mt':
            model = MarianMTModel.from_pretrained(os.path.join(model_dir,f"{translator}-{src_lang}-{tgt_lang}-{method}"))
        elif translator == 'mbart50':
            model = MBartForConditionalGeneration.from_pretrained(os.path.join(model_dir,f"{translator}-{src_lang}-{tgt_lang}-{method}"))
        else:
            raise ValueError
    
    # Load special tokenizer for method starting with averaged additional embeddings
    if method.find('-ae') != -1 or method.find('-re') != -1:
        if translator == 'opus-mt':
            tokenizer = MarianTokenizer.from_pretrained(os.path.join(tokenizer_dir,f"with_professions_{translator}-{src_lang}-{tgt_lang}/"))
        elif translator == "mbart50":
            tokenizer = MBart50TokenizerFast.from_pretrained(os.path.join(tokenizer_dir, f"with_professions_{translator}-{src_lang}-{tgt_lang}/"))
    elif translator == 'opus-mt' or (translator == 'new-opus-mt' and method.endswith('split-professions')):
        tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{tgt_lang}")
    elif translator == 'new-opus-mt' and method.endswith('keep-professions'):
        tokenizer = MarianTokenizer.from_pretrained(os.path.join(tokenizer_dir,f"with_professions_opus-mt-{src_lang}-{tgt_lang}/"))
    elif translator == "mbart50":
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang=LANGUAGE_CODES_MAP[src_lang], tgt_lang=LANGUAGE_CODES_MAP[tgt_lang])

    else:
        raise ValueError

    with open("../data/en.txt","r") as f:
        lines = f.readlines()
        lines = [l.split("\t")[2] for l in lines]
        
    os.makedirs(os.path.join(output_dir,f"{translator}-{method}"), exist_ok=True)
    print("path*******")
    print(os.path.join(output_dir,f"{translator}-{method}/{src_lang}-{tgt_lang}.txt"))
    with open(os.path.join(output_dir,f"{translator}-{method}/{src_lang}-{tgt_lang}.txt"),"w+") as f_out, \
         open(os.path.join(output_dir,f"{translator}-{method}/{tgt_lang}_num_sentences_with_added_tokens.txt"),"w") as f_out2:
        translated, n_sents_with_added_tokens = translate_sentences(lines, model, tokenizer)
        for orginal_sent, translated_sent in zip(lines, translated):
            f_out.write(orginal_sent+" ||| "+translated_sent+"\n")
        f_out2.write(str(n_sents_with_added_tokens)+"\n")

