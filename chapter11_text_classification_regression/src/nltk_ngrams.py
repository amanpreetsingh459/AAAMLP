# -*- coding: utf-8 -*-

from nltk import ngrams
from nltk.tokenize import word_tokenize
# let's see 3 grams
N = 3
# input sentence
sentence = "hi, how are you?"
# tokenized sentence
tokenized_sentence = word_tokenize(sentence)
# generate n_grams
n_grams = list(ngrams(tokenized_sentence, N))
print(n_grams)

