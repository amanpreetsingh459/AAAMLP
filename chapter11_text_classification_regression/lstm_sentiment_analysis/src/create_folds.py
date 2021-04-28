# -*- coding: utf-8 -*-

# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Read training data
    df = pd.read_csv("../input/imdb.csv")
    
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
                    lambda x: 1 if x == "positive" else 0
                    )
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch labels
    y = df.sentiment.values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
        
    # save the new csv with kfold column
    df.to_csv("../input/imdb_folds.csv", index=False)
    
    