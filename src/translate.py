from easynmt import EasyNMT
model = EasyNMT('opus-mt')
from tqdm import tqdm
with open("../data/en.txt", "r") as f:
    lines = f.readlines()
    lines = [l.split("\t")[2] for l in lines]
with open("../data/en-de","w+") as f_de, open("../data/en-he","w+") as f_he:
    for i in tqdm(range(len(lines))):
        l=lines[i]
        translated_de = model.translate(l,source_lang="en",target_lang="de")
        f_de.write(l+" ||| "+translated_de+"\n")
        translated_he = model.translate(l,source_lang="en",target_lang="he")
        f_he.write(l+" ||| "+translated_he+"\n")

