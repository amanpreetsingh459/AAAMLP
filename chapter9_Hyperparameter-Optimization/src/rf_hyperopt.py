# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space
    and training features and targets. It then initializes
    the models by setting the chosen parameters and runs
    cross-validation and returns a negative accuracy score
    :param params: dict of params from hyperopt
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)
    
    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=5)
        
    # initialize accuracy list
    accuracies = []
    
    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        
        xtest = x[test_idx]
        ytest = y[test_idx]
        
        # fit model for current fold
        model.fit(xtrain, ytrain)
        
        #create predictions
        preds = model.predict(xtest)
        
        # calculate and append accuracy
        fold_accuracy = metrics.accuracy_score(
                                        ytest,
                                        preds
                                        )
        
        accuracies.append(fold_accuracy)
        
    # return negative accuracy
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/mobile_train.csv")
    
    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df.drop("price_range", axis=1).values
    
    # and the targets
    y = df.price_range.values
    
    # define a parameter space
    # now we use hyperopt
    param_space = {
            # quniform gives round(uniform(low, high) / q) * q
            # we want int values for depth and estimators
            "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
            "n_estimators": scope.int(
                                hp.quniform("n_estimators", 100, 1500, 1)
                                ),
            # choice chooses from a list of values
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            
            # uniform chooses a value between two values
            "max_features": hp.uniform("max_features", 0, 1)
        }
    
    # partial function
    optimization_function = partial(
                                optimize,
                                x=X,
                                y=y
                                )
    
    # initialize trials to keep logging information
    trials = Trials()
    
    # run hyperopt
    hopt = fmin(
                fn=optimization_function,
                space=param_space,
                algo=tpe.suggest,
                max_evals=15,
                trials=trials
                )
    
    print(hopt)