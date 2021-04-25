# -*- coding: utf-8 -*-

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string


def clean_text(s):
    """
    This function cleans the text a bit
    :param s: string
    :return: cleaned string
    """
    
    # split by all whitespaces
    s = s.split()
    
    # join tokens by single space
    # why we do this?
    # this will remove all kinds of weird space
    # "hi. how are you" becomes
    # "hi. how are you"
    s = " ".join(s)
    
    # remove all punctuations using regex and string module
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)
    
    # you can add more cleaning here if you want
    # and then return the cleaned string
    
    return s


# create a corpus of sentences
# we read only 10k samples from training data
# for this example
corpus = pd.read_csv("../input/imdb.csv", nrows=10000)

corpus.loc[:, "review"] = corpus.review.apply(clean_text)

corpus = corpus.review.values

# initialize TfidfVectorizer with word_tokenize from nltk
# as the tokenizer
tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

# fit the vectorizer on corpus
tfv.fit(corpus)

# transform the corpus using tfidf
corpus_transformed = tfv.transform(corpus)

# initialize SVD with 10 components
svd = decomposition.TruncatedSVD(n_components=10)

# fit SVD
corpus_svd = svd.fit(corpus_transformed)

# choose first sample and create a dictionary
# of feature names and their scores from svd
# you can change the sample_index variable to
# get dictionary for any other sample
sample_index = 0

N = 5

for sample_index in range(5):
    feature_scores = dict(
        zip(
            tfv.get_feature_names(),
            corpus_svd.components_[sample_index]
            ))
                            
    # once we have the dictionary, we can now
    # sort it in decreasing order and get the
    # top N topics
    print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])
    

