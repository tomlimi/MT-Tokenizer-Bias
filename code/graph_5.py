import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean

RECALL=-3
PRECISION=-1


def graph_5_delta_g(results_file,tokens_file,lang):
    with open(results_file, "r") as f:
        lines = f.readlines()
        recalls_str = lines[RECALL]
        recalls_str = recalls_str.replace("'", "\"")
        recalls_dict = json.loads(recalls_str)

        precisions_str = lines[PRECISION]
        precisions_str = precisions_str.replace("'", "\"")
        precisions_dict = json.loads(precisions_str)

    with open(tokens_file, "r") as f:
        tokens_str = (f.readlines())[0]
        tokens_str = tokens_str.replace("'", "\"")
        tokens_dict = json.loads(tokens_str)

    if 'worker' in recalls_dict:
        recalls_dict.pop('worker')
    if 'worker' in precisions_dict:
        precisions_dict.pop('worker')
    if 'worker' in tokens_dict:
        tokens_dict.pop('worker')

    professions = list(tokens_dict.keys())
    delta_t_dict = dict()
    for p in professions:
        delta_t_dict[p] = tokens_dict[p]['Male'] - tokens_dict[p]['Female']
    delta_t_dict = dict(sorted(delta_t_dict.items(), key=lambda item: item[1]))

    delta_g_dict = dict()
    for p in delta_t_dict.keys():
        if p == 'CEO':
            p = 'ceo'
        r_m,r_f=recalls_dict[p]['male_recall'],recalls_dict[p]['female_recall']
        p_m,p_f=precisions_dict[p]['male_precision'],precisions_dict[p]['female_precision']
        if r_m ==0 and p_m == 0:
            male_f1 = 0
        else:
            male_f1=(r_m*p_m)/(r_m+p_m)

        if r_f ==0 and p_f == 0:
            female_f1 = 0
        else:
            female_f1=(r_f*p_f)/(r_f+p_f)
        delta_g_dict[p] = male_f1 - female_f1

    x = list(delta_t_dict.values())
    y = list(delta_g_dict.values())
    plt.scatter(x, y, marker='o')
    plt.xlabel("Delta T")
    plt.ylabel("Delta G")
    plt.title("Delta G per Delta T for each profession " + lang)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.show()
def graph_5(results_file,tokens_file,lang, recall_precision_location):
    if recall_precision_location==RECALL:
        male_key,female_key='male_recall','female_recall'
    else:
        male_key,female_key='male_precision','female_precision'

    with open(results_file,"r") as f:
        recalls_precisions_str = (f.readlines())[recall_precision_location]
        recalls_precisions_str = recalls_precisions_str.replace("'","\"")
        recalls_precisions_dict = json.loads(recalls_precisions_str)

    with open(tokens_file,"r") as f:
        tokens_str = (f.readlines())[0]
        tokens_str = tokens_str.replace("'","\"")
        tokens_dict = json.loads(tokens_str)

    if 'worker' in recalls_precisions_dict:
        recalls_precisions_dict.pop('worker')
    if 'worker' in tokens_dict:
        tokens_dict.pop('worker')

    professions = list(tokens_dict.keys())
    delta_t_dict = dict()
    for p in professions:
        delta_t_dict[p] = tokens_dict[p]['Male']-tokens_dict[p]['Female']
    delta_t_dict = dict(sorted(delta_t_dict.items(), key=lambda item: item[1]))

    recall_or_preciosnion_dict = dict()
    for p in delta_t_dict.keys():
        if p=='CEO':
           p='ceo'
        recall_or_preciosnion_dict[p] = recalls_precisions_dict[p][male_key]-recalls_precisions_dict[p][female_key]

    x = list(delta_t_dict.values())
    y = list(recall_or_preciosnion_dict.values())
    plt.scatter(x, y, marker='o')
    plt.xlabel("Delta T")
    if recall_precision_location == RECALL:
        plt.ylabel("Recall")
        plt.title("Recall per Delta T for each profession "+lang)
    else:
        plt.ylabel("Precision")
        plt.title("Precision per Delta T for each profession "+lang)


    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.show()

if __name__ == '__main__':
    graph_5("../data/he_results.txt", "../data/he_tokens_per_profession.txt","Hebrew",RECALL)
    graph_5("../data/de_results.txt", "../data/de_tokens_per_profession.txt","German",RECALL)
    graph_5("../data/he_results.txt", "../data/he_tokens_per_profession.txt","Hebrew",PRECISION)
    graph_5("../data/de_results.txt", "../data/de_tokens_per_profession.txt","German",PRECISION)
    graph_5_delta_g("../data/he_results.txt", "../data/he_tokens_per_profession.txt","Hebrew")
    graph_5_delta_g("../data/de_results.txt", "../data/de_tokens_per_profession.txt","German")