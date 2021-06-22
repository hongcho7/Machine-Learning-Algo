'''
df = data + target
'''

import pandas as pd
df = pd.read_csv("winequality-red.csv")

'''
random shuffle
- when you want to shuffle randomly
- add the code inside/outside the loop
'''
df = df.sample(frac=1).reset_index(drop=True)



'''
holdout validation
- also known as common splitting
'''

df = df.sample(frac=1).reset_index(drop=True)

df_train = df.head(1000)
df_test = df.tail(599)



'''
k-fold cross-validation
- split randomly before k-fold cross-validation
'''

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
# used to execute some code only if the file was run directly, and not imported
    df = pd.read_csv("train.csv")
    
    # create a new column and fill it with -1
    df['kfold'] = -1
    
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    kf = model_selection.KFold(n_splits=5)
    
    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
        
    df.to_csv("train_folds.csv", index = False)
    
    
    
    
'''
stratified k fold for classification
'''

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
# used to execute some code only if the file was run directly, and not imported
    df = pd.read_csv("train.csv")
    
    # create a new column and fill it with -1
    df['kfold'] = -1
    
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
        
    df.to_csv("train_folds.csv", index = False)




'''
using stratrified Kfold for regression
Step 1: Make Bins using Sturge Rule
Step 2: imply stratified Kfold
'''

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def stratified_kfolds_reg(data):
    data['kfold'] = -1
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    # cacluate the number of bins by Sturge Rule
    num_bins = int(np.round(1 + np.log2(len(data))))
    
    data.loc[:, 'bins'] = pd.cut(data['target'], bins =     num_bins, labels = False)
    
    kf = model_selection.StratifiedKfold(n_splits=5)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
        
        data = data.drop("bins", axis = 1)
        
    return data
        
        
