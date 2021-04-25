# -*- coding: utf-8 -*-

import io

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize

from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer


def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
                fname,
                'r',
                encoding='utf-8',
                newline='\n',
                errors='ignore'
            )
    
    n, d = map(int, fin.readline().split())
    
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        
    return data


def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    """
    Given a sentence and other information,
    this function returns embedding for the whole sentence
    :param s: sentence, string
    :param embedding_dict: dictionary word:vector
    :param stop_words: list of stop words, if any
    :param tokenizer: a tokenization function
    """
    
    # convert sentence to string and lowercase it
    words = str(s).lower()
    
    # tokenize the sentence
    words = tokenizer(words)
    
    # remove stop word tokens
    words = [w for w in words if not w in stop_words]
    
    # keep only alpha-numeric tokens
    words = [w for w in words if w.isalpha()]
    
    # initialize empty list to store embeddings
    M = []
    
    for w in words:
        # for every word, fetch the embedding from
        # the dictionary and append to list of
        # embeddings
        if w in embedding_dict:
            M.append(embedding_dict[w])
            
    # if we dont have any vectors, return zeros
    if len(M) == 0:
        return np.zeros(300)
    
    # convert list of embeddings to array
    M = np.array(M)
    
    # calculate sum over axis=0
    v = M.sum(axis=0)
    
    # return normalized vector
    return v / np.sqrt((v ** 2).sum())


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/imdb.csv")
    
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
                        lambda x: 1 if x == "positive" else 0
                        )
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # load embeddings into memory
    print("Loading embeddings")
    embeddings = load_vectors("../input/crawl-300d-2M.vec")
    
    # create sentence embeddings
    print("Creating sentence vectors")
    vectors = []
    for review in df.review.values:
        vectors.append(
            sentence_to_vec(
                s = review,
                embedding_dict = embeddings,
                stop_words = [],
                tokenizer = word_tokenize
            )
        )
        
    vectors = np.array(vectors)
    
    # fetch labels
    y = df.sentiment.values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for fold_, (t_, v_) in enumerate(kf.split(X=vectors, y=y)):
        print(f"Training fold: {fold_}")
        
        # temporary dataframes for train and test
        xtrain = vectors[t_, :]
        ytrain = y[t_]
        
        xtest = vectors[v_, :]
        ytest = y[v_]
        
        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, ytrain)
        
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        
        # calculate accuracy
        accuracy = metrics.accuracy_score(ytest, preds)
        print(f"Accuracy = {accuracy}")
        print("")