from transformers import MarianTokenizer
import numpy as np
import matplotlib.pyplot as plt
import collections
plt.style.use('seaborn-deep')

tokenizer_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tokenizer_he = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")

hebrew_file_names = ["../human_annotations/he1_translations", "../human_annotations/he2_translations",
                     "../human_annotations/he3_translations"]
german_file_names = ["../human_annotations/de1_translations", "../human_annotations/de2_translations",
                     "../human_annotations/de3_translations"]


def merge_translations(file_names, target_file):
    translations_dict = {}
    professions = set()
    for file in file_names:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            columns = line.split("\t")
            english_profession = columns[0]
            professions.add(english_profession)
            if not english_profession in translations_dict:
                translations_dict[english_profession] = {'Male': set(), 'Female': set()}
            for i in range(1, len(columns)):
                if columns[i] != "":
                    if i % 2 and columns[i]:
                        translations_dict[english_profession]['Male'].add(columns[i])
                    else:
                        translations_dict[english_profession]['Female'].add(columns[i])
    with open(target_file, 'w+') as f:
        f.write(str(translations_dict))
    return translations_dict, professions

def get_annotations_dict(merged_annotations):
    translations_dict = {}
    professions = set()
    with open(merged_annotations, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        columns = line.split("\t")
        english_profession = columns[0]
        professions.add(english_profession)
        if not english_profession in translations_dict:
            translations_dict[english_profession] = {'Male': [], 'Female': []}
        for i in range(1, len(columns)):
            if columns[i] != "":
                if i % 2 and columns[i]:
                    translations_dict[english_profession]['Male'].append(columns[i])
                else:
                    translations_dict[english_profession]['Female'].append(columns[i])
    return translations_dict, professions


def get_num_of_tokens_per_profession(professions, translations_dict, tokenizer, target_file):
    tokens_per_profession = {}
    for profession in professions:
        tokens_per_profession[profession] = {}
        male_count, male_tokens, female_count, female_tokens = 0, 0, 0, 0
        m, f = list(translations_dict[profession]['Male']), list(translations_dict[profession]['Female'])
        with tokenizer.as_target_tokenizer():
            for mp in m:
                male_count += 1
                male_tokens += len(tokenizer(mp)['input_ids'][:-1])
            tokens_per_profession[profession]['Male'] = male_tokens / male_count
            for fp in f:
                female_count += 1
                female_tokens += len(tokenizer(fp)['input_ids'][:-1])
        tokens_per_profession[profession]['Female'] = female_tokens / female_count
    with open(target_file, 'w+') as f:
        f.write(str(tokens_per_profession))
    return tokens_per_profession


def get_num_of_tokens_per_gender(professions, translations_dict, tokenizer, target_file):
    male_count, male_tokens, female_count, female_tokens = 0, 0, 0, 0
    male_num_of_tokens_map, female_num_of_tokens_map = {}, {}
    for profession in professions:
        m, f = list(translations_dict[profession]['Male']), list(translations_dict[profession]['Female'])
        with tokenizer.as_target_tokenizer():

            for mp in m:
                male_count += 1
                male_num_of_tokens = len(tokenizer(mp)['input_ids'][:-1])
                male_tokens += male_num_of_tokens
                if male_num_of_tokens not in male_num_of_tokens_map:
                    male_num_of_tokens_map[male_num_of_tokens] = 1
                else:
                    male_num_of_tokens_map[male_num_of_tokens] += 1
            for fp in f:
                female_count += 1
                female_num_of_tokens = len(tokenizer(fp)['input_ids'][:-1])
                female_tokens += female_num_of_tokens
                if female_num_of_tokens not in female_num_of_tokens_map:
                    female_num_of_tokens_map[female_num_of_tokens] = 1
                else:
                    female_num_of_tokens_map[female_num_of_tokens] += 1
    with open(target_file, 'w+') as f:
        f.write("male: " + str(male_tokens / male_count) + "\n")
        f.write("female: " + str(female_tokens / female_count) + "\n")
    max_tokens = max(max(male_num_of_tokens_map.keys()),max(female_num_of_tokens_map.keys()))
    return male_tokens / male_count, female_tokens / female_count, male_num_of_tokens_map, female_num_of_tokens_map, max_tokens

def count_tokens(translations_dict, mp, count,tokenizer, tokens, num_of_tokens_map, gender):
    professions = list(translations_dict[mp][gender])
    count += len(professions)
    for p in professions:
        num_of_tokens = len(tokenizer(p)['input_ids'][:-1])
        tokens += num_of_tokens
        if num_of_tokens not in num_of_tokens_map:
            num_of_tokens_map[num_of_tokens] = 1
        else:
            num_of_tokens_map[num_of_tokens] += 1
    return tokens,count
def get_num_of_tokens_per_stereotype(male_stereotype,female_stereotype,translations_dict, tokenizer, target_file):
    stereotype_count, stereotype_tokens, anti_stereotype_count, anti_stereotype_tokens = 0, 0, 0, 0
    stereotype_num_of_tokens_map, anti_stereotype_num_of_tokens_map = {}, {}
    with tokenizer.as_target_tokenizer():
        for mp in male_stereotype:
            tokens,count = count_tokens(translations_dict, mp, stereotype_count,tokenizer, stereotype_tokens, stereotype_num_of_tokens_map,'Male')
            stereotype_tokens+=tokens
            stereotype_count+=count
            tokens,count = count_tokens(translations_dict, mp, anti_stereotype_count,tokenizer, anti_stereotype_tokens, anti_stereotype_num_of_tokens_map,'Female')
            anti_stereotype_tokens += tokens
            anti_stereotype_count += count
        for fp in female_stereotype:
            tokens,count = count_tokens(translations_dict, fp, stereotype_count,tokenizer, stereotype_tokens, stereotype_num_of_tokens_map,'Female')
            stereotype_tokens += tokens
            stereotype_count += count
            tokens,count = count_tokens(translations_dict, fp, anti_stereotype_count,tokenizer, anti_stereotype_tokens, anti_stereotype_num_of_tokens_map,'Male')
            anti_stereotype_tokens += tokens
            anti_stereotype_count += count
    with open(target_file, 'w+') as f:
        f.write("stereotype: " + str(stereotype_tokens / stereotype_count) + "\n")
        f.write("anti stereotype: " + str(anti_stereotype_tokens / anti_stereotype_count) + "\n")
    max_tokens = max(max(stereotype_num_of_tokens_map.keys()),max(anti_stereotype_num_of_tokens_map.keys()))
    return stereotype_tokens / stereotype_count, anti_stereotype_tokens / anti_stereotype_count,\
           stereotype_num_of_tokens_map, anti_stereotype_num_of_tokens_map, max_tokens

def graphs_3_and_4(group1_num_of_tokens_map, group2_num_of_tokens_map, max_tokens, title, group1_name,group2_name):
    for i in range(1,max_tokens+1):
        if i not in group1_num_of_tokens_map:
            group1_num_of_tokens_map[i] = 0
        if i not in group2_num_of_tokens_map:
            group2_num_of_tokens_map[i] = 0
    group1_num_of_tokens_map = collections.OrderedDict(sorted(group1_num_of_tokens_map.items()))
    group2_num_of_tokens_map = collections.OrderedDict(sorted(group2_num_of_tokens_map.items()))
    grop1 = list(group1_num_of_tokens_map.values())
    grop2 = list(group2_num_of_tokens_map.values())

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))


    # Set position of bar on X axis
    br1 = np.arange(1,len(grop1)+1)
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, grop1, color='blue', width=barWidth,
            edgecolor='grey', label=group1_name)
    plt.bar(br2, grop2, color='pink', width=barWidth,
            edgecolor='grey', label=group2_name)

    # Adding Xticks
    plt.xlabel('num of tokens', fontweight='bold', fontsize=15)
    plt.ylabel('num of professions', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth/2 for r in range(1, len(grop1)+1)],range(1, len(grop1)+1))

    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    # this was in the case where we had all annotations and we wanted a sum of them
    # he_translations, professions = merge_translations(hebrew_file_names, "../data/he_merged_translations.txt")
    # de_translations, professions = merge_translations(german_file_names, "../data/de_merged_translations.txt")

    # this is in case we have one merged annotation file with the majority of 2 annotators and we create dict from it
    he_translations, professions = get_annotations_dict("../data/he_merged_translations_new.txt")
    de_translations, professions = get_annotations_dict("../data/de_merged_translations_new.txt")
    print("tokens per profession he")
    print(get_num_of_tokens_per_profession(professions, he_translations, tokenizer_he, "../data/he_tokens_per_profession.txt"))
    print("tokens per profession de")
    print(get_num_of_tokens_per_profession(professions, de_translations, tokenizer_de, "../data/de_tokens_per_profession.txt"))

    print("tokens per gender he")
    he_male_average_tokens, he_female_average_tokens, he_male_num_of_tokens_map, he_female_num_of_tokens_map, he_max_tokens = \
        get_num_of_tokens_per_gender(professions, he_translations, tokenizer_he, "../data/he_tokens_per_gender.txt")
    graphs_3_and_4(he_male_num_of_tokens_map, he_female_num_of_tokens_map, he_max_tokens, "Hebrew num of tokens per Gender", "Male", "Female")
    print("tokens per gender de")
    de_male_average_tokens, de_female_average_tokens, de_male_num_of_tokens_map, de_female_num_of_tokens_map, de_max_tokens = \
        get_num_of_tokens_per_gender(professions, de_translations, tokenizer_de, "../data/de_tokens_per_gender.txt")
    graphs_3_and_4(de_male_num_of_tokens_map, de_female_num_of_tokens_map, de_max_tokens, "German num of tokens per Gender", "Male", "Female")


    # stereotype

    with open("../data/male_stereotype","r") as f:
        male_stereotype = f.readlines()
        male_stereotype = [i.strip() for i in male_stereotype]
    with open("../data/female_stereotype","r") as f:
        female_stereotype = f.readlines()
        female_stereotype = [i.strip() for i in female_stereotype]

    print("tokens per stereotype he")
    he_stereotype_avg_tokens, he_anti_stereotype_avg_tokens, \
    he_stereotype_num_of_tokens_map, he_anti_stereotype_num_of_tokens_map, he_max_tokens = \
        get_num_of_tokens_per_stereotype(male_stereotype, female_stereotype, he_translations, tokenizer_he, "../data/he_tokens_per_stereotype.txt")
    graphs_3_and_4(he_stereotype_num_of_tokens_map, he_anti_stereotype_num_of_tokens_map, he_max_tokens, "Hebrew num of tokens per stereotype", "stereotype", "Anti stereotype")

    print("tokens per stereotype de")
    de_stereotype_avg_tokens, de_anti_stereotype_avg_tokens, \
    de_stereotype_num_of_tokens_map, de_anti_stereotype_num_of_tokens_map, de_max_tokens = \
        get_num_of_tokens_per_stereotype(male_stereotype, female_stereotype, de_translations, tokenizer_de, "../data/de_tokens_per_stereotype.txt")
    graphs_3_and_4(de_stereotype_num_of_tokens_map, de_anti_stereotype_num_of_tokens_map, de_max_tokens, "German num of tokens per stereotype", "stereotype", "Anti stereotype")
