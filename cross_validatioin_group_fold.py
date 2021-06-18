'''
stratified groupKFold
- folds are made by preserving the perecentage of sample for each class
- the same group will not appear in two different folds
'''

import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# defaultdict: 딕셔너리와 거의 비슷하지만 key값이 없을 경우 미리 지정해놓은 초기(default)값을 반환하는 dictionary
# Counter: 컨테이너에 동일한 값의 자료가 몇개인지를 파악하는데 사용하는 객체

def stratified_group_k_fold(X, y, groups, k, seed = None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    
    for label, g in zip (y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1
        
    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num)
    groups_per_fold = defaultdict(set)
    
    def eval_y_counts_per_fold(y_counts, fold):
    
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            
            std_per_label.append(label_std)
        
        y_counts_per_fold -= y_counts
        return np.mean(std_per_label)
        
        
        
    
        
    
