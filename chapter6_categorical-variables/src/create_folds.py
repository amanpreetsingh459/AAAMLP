# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
    # Read training data
    df = pd.read_csv("../input/cat_train.csv")
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch labels
    y = df.target.values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    # save the new csv with kfold column
    df.to_csv("../input/cat_train_folds.csv", index=False)
    