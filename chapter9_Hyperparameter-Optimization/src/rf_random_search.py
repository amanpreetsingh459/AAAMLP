# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/mobile_train.csv")
    # features are all columns without price_range
    # note that there is no id column in this dataset
    
    # here we have training features
    X = df.drop("price_range", axis=1).values
    
    # and the targets
    y = df.price_range.values
    
    # define the model here
    # i am using random forest with n_jobs=-1
    # n_jobs=-1 => use all cores
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # define a grid of parameters
    # this can be a dictionary or a list of
    # dictionaries
    param_grid = {
    "n_estimators": np.arange(100, 1500, 100),
    "max_depth": np.arange(1, 31),
    "criterion": ["gini", "entropy"]
    }
    
    # initialize random search
    # estimator is the model that we have defined
    # param_distributions is the grid/distribution of parameters
    # we use accuracy as our metric. you can define your own
    # higher value of verbose implies a lot of details are printed
    # cv=5 means that we are using 5 fold cv (not stratified)
    # n_iter is the number of iterations we want
    # if param_distributions has all the values as list,
    # random search will be done by sampling without replacement
    # if any of the parameters come from a distribution,
    # random search uses sampling with replacement
    model = model_selection.RandomizedSearchCV(
                                estimator=classifier,
                                param_distributions=param_grid,
                                n_iter=20,
                                scoring="accuracy",
                                verbose=10,
                                n_jobs=1,
                                cv=5
                                )
    # fit the model and extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
        
        

