import json
import matplotlib.pyplot as plt
import numpy as np
def graph_5(recall_file,tokens_file,lang):
    with open(recall_file,"r") as f:
        recalls_str = (f.readlines())[-1]
        recalls_str = recalls_str.replace("'","\"")
        recalls_dict = json.loads(recalls_str)
    with open(tokens_file,"r") as f:
        tokens_str = (f.readlines())[0]
        tokens_str = tokens_str.replace("'","\"")
        tokens_dict = json.loads(tokens_str)

    professions = list(tokens_dict.keys())
    delta_t_dict = dict()
    for p in professions:
        delta_t_dict[p] = tokens_dict[p]['Male']-tokens_dict[p]['Female']

    delta_t_dict = dict(sorted(delta_t_dict.items(), key=lambda item: item[1]))

    delta_g_dict = dict()
    for p in delta_t_dict.keys():
        delta_g_dict[p] = recalls_dict[p]['male_recall']-recalls_dict[p]['female_recall']

    x = list(delta_t_dict.values())
    y = list(delta_g_dict.values())
    plt.scatter(x, y, marker='o')
    plt.xlabel("Delta T")
    plt.ylabel("Delta G")
    plt.title("Delta G per Delta T for each profession "+lang)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.show()

if __name__ == '__main__':
    graph_5("../data/he_recall_per_profession.txt", "../data/he_tokens_per_profession.txt","Hebrew")
    graph_5("../data/de_recall_per_profession.txt", "../data/de_tokens_per_profession.txt","German")