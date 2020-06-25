from sklearn.model_selection import GroupKFold, KFold
import pandas as pd
import numpy as np

class ShuffleGroupKFold():
    """
    K-fold iterator variant with non-overlapping groups. The groups are initially shuffled.

    The same group will not appear in two different folds (the number of distinct groups has to be at least equal to the number of folds).

    The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold.
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def split(self, group: pd.Series):
        
        gp_label = group.unique()
        gp_cv = KFold(self.n_splits, shuffle=True)

        split = []
        
        for gp_train_index, gp_test_index in gp_cv.split(gp_label):
            
            gp_train = gp_label[gp_train_index]
            train_index = np.argwhere(group.isin(gp_train).values)[:,0]
            
            gp_test = gp_label[gp_test_index]
            test_index = np.argwhere(group.isin(gp_test).values)[:,0]
            
            split.append((train_index, test_index))
            
        return split
            
            
            