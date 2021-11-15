import numpy as np
import pandas as pd
import scipy.stats as ss
import sys
import os
import time
# import utils
## read data
from utils.data_loader import DataLoader
from utils.build_classifiers import Classifiers
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from IPython.display import display


# Main idea: 一个外套走完所有数据集
# read data
def load_data():
    dataloader = DataLoader()
    train_X, train_y = dataloader.get_train_data()
    return train_X, train_y

def filter_big_data(train_X, train_y):
    a = []
    b = []
    c = []
    for fname, data in train_X.items():
        df_tmp = pd.DataFrame(data)
        a.append(fname)
        b.append(round(df_tmp.memory_usage(deep=True).sum() * 1e-6, 4))
        c.append(str(round(sum(train_y[fname]) / len(train_y[fname]) * 100, 4)) + "%")
    df_discription = pd.DataFrame({"FileName": a, "Memory_in_MB": b, "OutliersRate": c})
    small_sets = set(df_discription.FileName[df_discription.Memory_in_MB <= 5])
    new_train_X = dict()
    new_train_y = dict()
    for fname, data in train_X.items():
        if fname in small_sets:
            new_train_X[fname] = train_X[fname]
            new_train_y[fname] = train_y[fname]
    return new_train_X, new_train_y


def train_classifiers():
    x, y = load_data()
    training_results = dict()
    train_X, train_y = filter_big_data(x,y)
    normalizer = RobustScaler
    score_dfs = dict()
    clf_initializer = Classifiers("lofs")
    main_detectors = dict()
    performances = dict()
    mp.freeze_support()
    with mp.Pool(10) as pool:
        # files, data
        # data is an numpy ndarray
        for fname, data in train_X.items():
            # base classifiers
            training_results[fname] = pool.apply_async(clf_initializer.train_helper, (data, train_y[fname], normalizer))
        pool.close()
        pool.join()
        for key in training_results:
            training_results[key] = training_results[key].get(timeout=5)
    return training_results

def get_disagreement():
    pass

def get_thresholders():
    pass

def get_construct_methods():
    pass




if __name__ == "__main__":
    base_results = train_classifiers()