import numpy as np
import pandas as pd
import scipy.stats as ss
import sys
import os
import time
# import utils
## read data
from .data_loader import DataLoader
from .build_classifiers import Classifiers
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from IPython.display import display
from .calculate_disagreement import Disagreement
from .rank_score_diff import RankScoreDiff
from .threshold_method import *
from .construct_method import Constructor
from sklearn import metrics


# Main idea: 一个外套走完所有数据集
# read data
def load_data():
    dataloader = DataLoader()
    train_X, train_y = dataloader.get_train_data()
    return train_X, train_y


def filter_big_data(train_X, train_y, MB):
    a = []
    b = []
    c = []
    for fname, data in train_X.items():
        df_tmp = pd.DataFrame(data)
        a.append(fname)
        b.append(round(df_tmp.memory_usage(deep=True).sum() * 1e-6, 4))
        c.append(str(round(sum(train_y[fname]) / len(train_y[fname]) * 100, 4)) + "%")
    df_discription = pd.DataFrame({"FileName": a, "Memory_in_MB": b, "OutliersRate": c})
    small_sets = set(df_discription.FileName[df_discription.Memory_in_MB <= MB])
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
    train_X, train_y = filter_big_data(x, y, 60)
    normalizer = RobustScaler # RobustScaler
    score_dfs = dict()
    clf_initializer = Classifiers('mixed')
    main_detectors = dict()
    performances = dict()
    mp.freeze_support()
    with mp.Pool(10) as pool:
        # files, data
        # data is an numpy ndarray
        for fname, data in train_X.items():
            # base classifiers
            training_results[fname] = pool.apply_async(clf_initializer.train_helper, (data, train_y[fname], normalizer, fname))
        pool.close()
        pool.join()
        for key in training_results:
            training_results[key] = training_results[key].get(timeout=5)
    return training_results


def get_disagreement(score_df, rsdiff_obj):
    if type(score_df) == pd.DataFrame:
        score_ndarray = score_df.to_numpy()
    else:
        score_ndarray = score_df
    dis_obj = Disagreement(score_ndarray)
    disagreement_df = dis_obj.calc_disagreement(rsdiff_obj.get_rsd_matrix_2d())
    return disagreement_df


def get_rsd_obj(score_nd_array):
    tmp_var = RankScoreDiff(score_nd_array)
    rsdiff_obj = tmp_var.generate_rank_score_diff_property()
    return rsdiff_obj


def get_thresholders(disagreement_nd_array, zeros_like=False):
    s = np.apply_along_axis(get_thresholder_results, 0, disagreement_nd_array, zeros_like=zeros_like)
    return s


def get_construct_methods(score_df, main_detector_name, rsd_obj, disagreement_result_ndarray, flatten_columns):
    constructor = Constructor(score_df, main_detector_name, rsd_obj, disagreement_result_ndarray)
    constructor.predict(flatten_columns)
    return constructor.result_flatten_df