# -*- coding: utf-8 -*-

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    df = pd.read_csv("../input/mnist_train_folds.csv")
    
    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    
    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values
    
    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    
    # fit the model on training data
    clf.fit(x_train, y_train)
    
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    
    # save the model
    joblib.dump(clf, f"../models/dt_{fold}.bin")
    
if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)