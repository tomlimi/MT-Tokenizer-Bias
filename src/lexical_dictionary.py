import argparse
from collections import defaultdict


class LexicalDictionary:
    
    def __init__(self, dictionary_file):
        
        self.dictionary = defaultdict(set)
        self.dictionary_rev = defaultdict(set)
        self.src_vocab = set()
        self.tgt_vocab = set()
        self.load_dictionary(dictionary_file)
        
    def load_dictionary(self, dictionary_file):
       with open(dictionary_file, 'r') as in_s:
            for line in in_s:
                line = line.strip()
                if '\t' in line:
                    src_word, tgt_word = line.split("\t")
                else:
                    src_word, tgt_word = line.split(" ")
                
                self.src_vocab.add(src_word)
                self.tgt_vocab.add(tgt_word)
                self.dictionary[src_word].add(tgt_word)
                self.dictionary_rev[tgt_word].add(src_word)
                
    def __getitem__(self, word):
        return self.dictionary[word]
    
    def in_src(self, word):
        return word in self.src_vocab
    
    def in_tgt(self, word):
        return word in self.tgt_vocab