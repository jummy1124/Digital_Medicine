# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:20:22 2021

@author: 黃乾哲
"""
import nltk
import nltk.corpus
from nltk.text import Text
from nltk import word_tokenize
import jieba
import re 
import string
from contraction import CONTRACTION_MAP

extra_stop_word={"or",
                 "and",
                 "treatmentsprocedur",
                 "in",
                 "on",
                 "at",
                 "oper",
                 "other",
                 "gm",
                 "am",
                 "ms",
                 "day",
                 "year",
		 "type",
		 "cri"}

def content_fraction(text):
    text = decontracted(text)    
    text = text.lower()
    
    # remove punc
    text_remove     = '@\S+|https?:\S+|http?:\S' #|[^A-Za-z0-9]+
    text            = re.sub(text_remove,' ', text)
    text            = re.sub(r"[0-9]+", "", text)
    PUNCT_TO_REMOVE = string.punctuation
    text         = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    
    #Sentence segmentation
    word_list = nltk.word_tokenize(text)
    
    # stop word
    stops_list = set(nltk.corpus.stopwords.words('english'))
    stops_list.remove("no")
    stops_list = stops_list.union(extra_stop_word)
    
    tmp_word_list = word_list
    word_list=[]
    for word in tmp_word_list:    
        if word not in stops_list or re.search("not|n't", word):
            word_list.append(word)
    
    # stemming
    porter = nltk.stem.porter.PorterStemmer()
    tmp_word_list=word_list
    word_list=[]
    for word in tmp_word_list:        
        word=porter.stem(word)
        if len(word)>2 or word=="no":
            word_list.append(word)
    
    return word_list

def decontracted(phrase):
    
    for key in CONTRACTION_MAP.keys():
        phrase = re.sub(key, CONTRACTION_MAP[key], phrase)

    return phrase

if __name__ == "__main__":
    text = "123 24 Helllo I don't known. or on no a"
    print(content_fraction(text))

