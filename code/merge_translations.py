from transformers import MarianTokenizer
tokenizer_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tokenizer_he = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")

hebrew_file_names=["../human_annotations/he1_translations","../human_annotations/he2_translations","../human_annotations/he3_translations"]
german_file_names=["../human_annotations/de1_translations","../human_annotations/de2_translations","../human_annotations/de3_translations"]


def merge_translations(file_names,target_file):
    translations_dict={}
    professions=set()
    for file in file_names:
        with open(file,'r') as f:
            lines=f.readlines()
        for line in lines:
            line=line.strip()
            columns = line.split("\t")
            english_profession=columns[0]
            professions.add(english_profession)
            if not english_profession in translations_dict:
                translations_dict[english_profession] = {'Male':set(),'Female':set()}
            for i in range(1,len(columns)):
                if columns[i]!="":
                    if i%2 and columns[i]:
                        translations_dict[english_profession]['Male'].add(columns[i])
                    else:
                        translations_dict[english_profession]['Female'].add(columns[i])
    with open(target_file,'w+') as f:
        f.write(str(translations_dict))
    return translations_dict,professions

def get_num_of_tokens_per_profession(professions,translations_dict,tokenizer, target_file):
    tokens_per_profession = {}
    for profession in professions:
        tokens_per_profession[profession] = {}
        male_count,male_tokens,female_count,female_tokens=0,0,0,0
        m,f  =list(translations_dict[profession]['Male']),list(translations_dict[profession]['Female'])
        for mp in m:
            male_count+=1
            male_tokens+=len(tokenizer.tokenize(mp))
        tokens_per_profession[profession]['Male'] = male_tokens/male_count
        for fp in f:
            female_count+=1
            female_tokens+=len(tokenizer.tokenize(fp))
        tokens_per_profession[profession]['Female'] = female_tokens/female_count
    with open(target_file,'w+') as f:
        f.write(str(tokens_per_profession))
    return tokens_per_profession

def get_num_of_tokens_per_gender(professions,translations_dict,tokenizer,target_file):
    male_count,male_tokens,female_count,female_tokens=0,0,0,0
    for profession in professions:
        m,f  =list(translations_dict[profession]['Male']),list(translations_dict[profession]['Female'])
        for mp in m:
            male_count+=1
            male_tokens+=len(tokenizer.tokenize(mp))
        for fp in f:
            female_count+=1
            female_tokens+=len(tokenizer.tokenize(fp))
    with open(target_file,'w+') as f:
        f.write("male: "+str(male_tokens/male_count)+"\n")
        f.write("female: "+str(female_tokens/female_count)+"\n")
    return male_tokens/male_count,female_tokens/female_count





if __name__ == '__main__':
    he_translations,professions = merge_translations(hebrew_file_names, "he_merged_translations.txt")
    de_translations,professions = merge_translations(german_file_names, "de_merged_translations.txt")
    print("tokens per profession he")
    print(get_num_of_tokens_per_profession(professions,he_translations,tokenizer_he, "he_tokens_per_profession.txt"))
    print("tokens per profession de")
    print(get_num_of_tokens_per_profession(professions,de_translations,tokenizer_de, "de_tokens_per_profession.txt"))
    print("tokens per gender he")
    print(get_num_of_tokens_per_gender(professions,he_translations, tokenizer_he, "he_tokens_per_gender.txt"))
    print("tokens per gender de")
    print(get_num_of_tokens_per_gender(professions,de_translations, tokenizer_de, "de_tokens_per_gender.txt"))