import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean
from transformers import MarianTokenizer
from collections import defaultdict

RECALL=-3
PRECISION=-1

TRANSLATOR="opus-mt"
SRC_LANG="en"


def load_old_tokens_dict(tokens_file):
    with open(tokens_file, "r") as f:
        tokens_str = (f.readlines())[0]
    tokens_str = tokens_str.replace("'", "\"")
    tokens_dict = json.loads(tokens_str)
    
    if 'worker' in tokens_dict:
        tokens_dict.pop('worker')
        
    return tokens_dict


def create_tokens_dict(variants_file, translator, src_lang, tgt_lang):
    tokens_dict = defaultdict(dict)
    tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/{translator}-{src_lang}-{tgt_lang}")
    
    with open(variants_file, 'r') as var_json:
        variants_dict = json.load(var_json)
    
    for prof_gender, variants in variants_dict.items():
        prof, gender = prof_gender.split('-')
        gender = gender.capitalize()
        for v_idx, var in enumerate(variants):
            with tokenizer.as_target_tokenizer():
                tokenized = tokenizer(var)
                num_tokens = len(tokenized['input_ids']) - 1
            
            tokens_dict[f"{prof}-{str(v_idx)}"][gender] = num_tokens
    return tokens_dict


def graph_5_delta_g(results_file,tokens_dict,lang):
    with open(results_file, "r") as f:
        lines = f.readlines()
        recalls_str = lines[RECALL]
        recalls_str = recalls_str.replace("'", "\"")
        recalls_dict = json.loads(recalls_str)

        precisions_str = lines[PRECISION]
        precisions_str = precisions_str.replace("'", "\"")
        precisions_dict = json.loads(precisions_str)

    professions = list(tokens_dict.keys())
    delta_t_dict = dict()
    
    for p in professions:
        delta_t_dict[p] = tokens_dict[p]['Male'] - tokens_dict[p]['Female']
    delta_t_dict = dict(sorted(delta_t_dict.items(), key=lambda item: item[1]))


    x = []
    y = []

    for prof, delta_t in delta_t_dict.items():
        prof = prof.lower()
        if prof in recalls_dict and prof in precisions_dict:
            r_m,r_f=recalls_dict[prof]['male_recall'],recalls_dict[prof]['female_recall']
            p_m,p_f=precisions_dict[prof]['male_precision'],precisions_dict[prof]['female_precision']
            if r_m ==0 and p_m == 0:
                male_f1 = 0
            else:
                male_f1= 2 * (r_m*p_m)/(r_m+p_m)
    
            if r_f ==0 and p_f == 0:
                female_f1 = 0
            else:
                female_f1= 2 * (r_f*p_f)/(r_f+p_f)
                
            x.append(delta_t)
            y.append(male_f1 - female_f1)

    plt.scatter(x, y, marker='o', alpha=0.5)

    x_org = sorted(list(set(x)))
    y_org = [[] for _ in set(x)]
    
    for xi, yi in zip(x, y):
        box_i = x_org.index(xi)
        y_org[box_i].append(yi)
    
    plt.boxplot(y_org, positions=x_org)
    plt.xlabel("Delta T")
    plt.ylabel("Delta G")
    plt.title("Delta G per Delta T for each profession " + lang)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")
    #plt.ylim([-.2, .7])
    plt.savefig(f'../graphs/graph_5_{lang}_delta_g.pdf')
    plt.show()


def graph_5(results_file,tokens_dict,lang, recall_precision_location):
    if recall_precision_location==RECALL:
        male_key,female_key='male_recall','female_recall'
    else:
        male_key,female_key='male_precision','female_precision'

    with open(results_file,"r") as f:
        recalls_precisions_str = (f.readlines())[recall_precision_location]
        recalls_precisions_str = recalls_precisions_str.replace("'","\"")
        recalls_precisions_dict = json.loads(recalls_precisions_str)

    professions = list(tokens_dict.keys())
    delta_t_dict = dict()
    for p in professions:
        delta_t_dict[p] = tokens_dict[p]['Male']-tokens_dict[p]['Female']
    delta_t_dict = dict(sorted(delta_t_dict.items(), key=lambda item: item[1]))
    
    x = []
    y = []
    
    for prof, delta_t in delta_t_dict.items():
        prof = prof.lower()
        if prof in recalls_precisions_dict:
            x.append(delta_t)
            y.append(recalls_precisions_dict[prof][male_key]-recalls_precisions_dict[prof][female_key])

    # x = list(delta_t_dict.values())
    # y = list(recall_or_preciosnion_dict.values())
    plt.scatter(x, y, marker='o', alpha=0.5)
    x_org = sorted(list(set(x)))
    y_org = [[] for _ in set(x)]
    
    for xi, yi in zip(x, y):
        box_i = x_org.index(xi)
        y_org[box_i].append(yi)
    
    plt.boxplot(y_org, positions=x_org)
    plt.xlabel("Delta T")
    if recall_precision_location == RECALL:
        plt.ylabel("Recall")
        plt.title("Recall per Delta T for each profession "+lang)
    else:
        plt.ylabel("Precision")
        plt.title("Precision per Delta T for each profession "+lang)


    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    #plt.ylim([-.2, .7])
    plt.plot(x, p(x), "r--")

    if recall_precision_location == RECALL:
        plt.savefig(f'../graphs/graph_5_{lang}_recall.pdf')
    else:
        plt.savefig(f'../graphs/graph_5_{lang}_precision.pdf')

    plt.show()

if __name__ == '__main__':
    
    tokens_dict_he = create_tokens_dict("../data/wino_mt/he_variants.json",TRANSLATOR, SRC_LANG, "he")
    tokens_dict_de = create_tokens_dict("../data/wino_mt/de_variants.json",TRANSLATOR, SRC_LANG, "de")

    # tokens_dict_he = load_old_tokens_dict("../data/he_tokens_per_profession.txt")
    # tokens_dict_de = load_old_tokens_dict("../data/de_tokens_per_profession.txt")
    
    graph_5("../data/he_results.txt", tokens_dict_he,"Hebrew",RECALL)
    graph_5("../data/de_results.txt", tokens_dict_de,"German",RECALL)
    graph_5("../data/he_results.txt", tokens_dict_he,"Hebrew",PRECISION)
    graph_5("../data/de_results.txt", tokens_dict_de,"German",PRECISION)
    graph_5_delta_g("../data/he_results.txt", tokens_dict_he,"Hebrew")
    graph_5_delta_g("../data/de_results.txt", tokens_dict_de,"German")