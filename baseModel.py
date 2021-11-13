from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import re
from sklearn.preprocessing import RobustScaler
import scipy.stats as ss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import scipy.io
import os
import sys
from time import time
import scipy.stats as ss
from sklearn.preprocessing import RobustScaler
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from IPython.display import display

# dataset path
data_path = "/Users/kadima/experiment_any/anomaly-detection/datasets_resend/"


training_dict = dict()
label_dict = dict()

for root, path, files in os.walk("./datasets_resend/"):
    for file in files:
        if file.endswith("txt"):
            if "label" not in file:
                with open(root+file,'r') as d:
                    data = d.readlines()
                    data = [x.split() for x in data]
                    data = [[float(i) for i in x] for x in data]
                training_dict[file] = data
            else:
                with open(root+file,'r') as d:
                    label = d.readlines()
                    label = [int(x[0]) for x in label ]
                label_dict[file] = label


def sd_thresholder(scores, real_y, factor=2.5):
    high_limit = np.mean(scores) + factor * np.std(scores)
    y_predict = scores >= high_limit
    y_predict = [1 if j else 0 for j in y_predict]
    f1 = f1_score(real_y, y_predict)
    return y_predict, f1, high_limit


def mad_thresholder(scores, real_y):
    median_ = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - median_))
    y_predict = scores >= 3 * mad
    y_predict = [1 if j else 0 for j in y_predict]
    f1 = f1_score(real_y, y_predict)
    return y_predict, f1, 3 * mad


def iqr_thresholder(scores, real_y):
    iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
    y_predict = scores >= (np.percentile(scores, 75) + 1.5 * iqr)
    y_predict = [1 if j else 0 for j in y_predict]
    f1 = f1_score(real_y, y_predict)
    return y_predict, f1, np.percentile(scores, 75) + 1.5 * iqr


# the disagreement_v1 removes min rank and max rank in disagreement matrix and get the row's standard deviation.
# the simplest way
def disagreement_v1(score_matrix, num_detectors, real_y):
    rank_matrix = np.zeros([len(score_matrix), num_detectors])
    for i in range(num_detectors):
        # rank by each column and get its rank_position
        rank_matrix[:, i] = ss.rankdata(score_matrix[:, i])
    std_record = np.zeros(len(X))
    rank_record = []
    for i in range(len(score_matrix)):
        rank_rows = rank_matrix[i, :]
        min_ = np.min(rank_rows)
        max_ = np.max(rank_rows)
        rank_rows = [x for x in rank_rows if x not in [min_, max_]]
        std_record[i] = np.std(rank_rows)
        rank_record.append(rank_rows)
    valid_rank_rows = []
    for i in range(len(score_matrix)):
        num_large = np.sum(np.array(rank_record[i]) >= len(score_matrix) // 3)
        if num_large >= num_detectors // 3:
            valid_rank_rows.append(i)
    std_max_rows = np.argsort(np.array(std_record)[valid_rank_rows])[-1:]
    std_median_scores = np.median(score_matrix[std_max_rows, :], axis=0)
    threshold_for_each_detector = std_median_scores

    f1_list = []
    for i in range(num_detectors):
        outliers_rows = score_matrix[:, i] >= threshold_for_each_detector[i]
        y_predict = [1 if j else 0 for j in outliers_rows]
        f1 = f1_score(real_y, y_predict)
        f1_list.append(f1)
    return f1_list, threshold_for_each_detector, std_record


# 取 disagreement matrix 的std
def disagreement_v2(score_matrix, num_detectors, real_y, remove_extreme=False):
    # normalize the score_matrix
    origin_score_matrix = score_matrix.copy()
    score_matrix = RobustScaler().fit_transform(score_matrix)

    # get rank matrix
    rank_matrix = np.zeros([len(score_matrix), num_detectors])
    for i in range(num_detectors):
        # rank by each column and get its rank_position
        rank_matrix[:, i] = ss.rankdata(score_matrix[:, i], 'ordinal')
    std_record = np.zeros(len(score_matrix))

    # form a matrix for each row
    for row_idx in range(len(score_matrix)):
        tmp_matrix = np.zeros([num_detectors, num_detectors])
        for col_idx in range(num_detectors):
            rank_refer = rank_matrix[row_idx, col_idx]
            for col_idx_2 in range(num_detectors):
                if col_idx_2 == col_idx:
                    tmp_matrix[col_idx, col_idx_2] = 0
                else:
                    target_row = np.argwhere(rank_matrix[:, col_idx_2] == rank_refer)
                    target_row = target_row[0][0]
                    curr_score = score_matrix[row_idx, col_idx_2]
                    refer_score = score_matrix[target_row, col_idx_2]
                    tmp_matrix[col_idx, col_idx_2] = curr_score - refer_score

        if remove_extreme:
            tmp_matrix_row_std = np.std(tmp_matrix, axis=1)
            tmp_matrix_col_std = np.std(tmp_matrix, axis=0)
            max_std_row = np.argmax(tmp_matrix_row_std)
            max_std_col = np.argmax(tmp_matrix_col_std)
            # skip the max one
            robust_tmp_matrix = tmp_matrix[[i_ for i_ in range(num_detectors) if i_ != max_std_row], :]
            robust_tmp_matrix = robust_tmp_matrix[:, [i_ for i_ in range(num_detectors) if i_ != max_std_col]]


        else:
            robust_tmp_matrix = tmp_matrix

        std_record[row_idx] = np.mean(abs(robust_tmp_matrix))

    std_max_row = np.argmax(std_record)
    threshold_for_each_detector = score_matrix[std_max_row, :]
    origin_threshold = origin_score_matrix[std_max_row, :]
    f1_list = []

    for i in range(num_detectors):
        outliers_rows = score_matrix[:, i] >= threshold_for_each_detector[i]
        y_predict = [1 if j else 0 for j in outliers_rows]
        f1 = f1_score(real_y, y_predict)
        f1_list.append(f1)

    return f1_list, origin_threshold, std_record


# 取detector 自己的 disagreement
def disagreement_v3(score_matrix, num_detectors, real_y, remove_extreme=False):
    # normalize the score_matrix
    score_matrix = RobustScaler().fit_transform(score_matrix)

    # get rank matrix
    rank_matrix = np.zeros([len(score_matrix), num_detectors])
    for i in range(num_detectors):
        # rank by each column and get its rank_position
        rank_matrix[:, i] = ss.rankdata(score_matrix[:, i], 'ordinal')
    std_record = np.zeros((len(score_matrix), num_detectors))

    # form a matrix for each row
    for row_idx in range(len(score_matrix)):
        tmp_matrix = np.zeros([num_detectors, num_detectors])
        for col_idx in range(num_detectors):
            rank_refer = rank_matrix[row_idx, col_idx]
            col_increment = 0
            for col_idx_2 in range(num_detectors):
                target_row = np.argwhere(rank_matrix[:, col_idx_2] == rank_refer)
                target_row = target_row[0][0]
                curr_score = score_matrix[row_idx, col_idx_2]
                refer_score = score_matrix[target_row, col_idx_2]
                tmp_matrix[col_idx, col_increment] = curr_score - refer_score
                col_increment += 1

        if remove_extreme:
            tmp_matrix_row_std = np.std(tmp_matrix, axis=1)
            tmp_matrix_col_std = np.std(tmp_matrix, axis=0)
            max_std_row = np.argmax(tmp_matrix_row_std)
            max_std_col = np.argmax(tmp_matrix_col_std)
            # skip the max one
            robust_tmp_matrix = tmp_matrix[[i_ for i_ in range(num_detectors) if i_ != max_std_row], :]
            robust_tmp_matrix = robust_tmp_matrix[:, [i_ for i_ in range(num_detectors) if i_ != max_std_col]]
        else:
            robust_tmp_matrix = tmp_matrix

        detector_disagreement = []
        for j_ in range(num_detectors):
            std_record[row_idx, j_] = np.mean(np.abs(np.concatenate((robust_tmp_matrix[j_, :],
                                                                     robust_tmp_matrix[:, j_]),
                                                                    axis=None)))

    std_max_row = np.argmax(std_record, axis=0)
    threshold_for_each_detector = score_matrix[std_max_row, np.arange(num_detectors)]
    f1_list = []

    for i in range(num_detectors):
        outliers_rows = score_matrix[:, i] >= threshold_for_each_detector[i]
        y_predict = [1 if j else 0 for j in outliers_rows]
        f1 = f1_score(real_y, y_predict)
        f1_list.append(f1)

    return f1_list, threshold_for_each_detector, std_record


def get_score_matrix(X, num_detectors):
    return np.zeros([X.shape[0], num_detectors])

def get_perform_matrix(num_thresholders, num_detectors):
    return np.zeros((num_thresholders, num_detectors))


random_state = np.random.RandomState(10)
outliers_fraction = 0.4


# initialize a set of detectors for LSCP
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(contamination=outliers_fraction),
    # 'Cluster-based Local Outlier Factor (CBLOF)':
    #     CBLOF(contamination=outliers_fraction,
    #           check_estimator=False, random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state, n_estimators=280),
    'K Nearest Neighbors (KNN)': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Principal Component Analysis (PCA)': PCA(
        contamination=outliers_fraction, random_state=random_state),
    "COPOD": COPOD()
}

names = []
# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    names.append(clf)

# Fit the models with the generated data and
# compare model performances
def get_result(X, y, classifiers):
    threshold_records = list()
    # create matrix to store the performance
    score_matrix = get_score_matrix(X, len(classifiers.keys()))
    perform_table = get_perform_matrix(5, len(classifiers.keys()))

    np.random.seed(5)
    clfs = []
    # Fit the model
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        #         print(i + 1, 'fitting', clf_name)
        # fit the data and tag outliers
        print(clf_name)
        clf.fit(X)
        clfs.append(clf)
        scores_pred = clf.decision_function(X)
        score_matrix[:, i] = scores_pred

    for i, thresholder in enumerate([sd_thresholder, mad_thresholder,
                                     iqr_thresholder]):
        kk = []
        for j in range(score_matrix.shape[1]):
            _, perform_table[i, j], b = thresholder(score_matrix[:, j], y)
            kk.append(b)

        threshold_records.append(kk)

    for i in range(score_matrix.shape[1]):
        perform_table[-2, i] = f1_score(y, clfs[i].predict(X))

    perform_table[-1, :], a, dist_ = disagreement_v2(score_matrix, len(classifiers), y, True)
    threshold_records.append(a)

    return (pd.DataFrame(perform_table, columns=names, index=["sd", 'mad', 'iqr', 'default', 'disagreement']),
            threshold_records, dist_, score_matrix)

result_dict = dict()

for data_name, X in training_dict.items():
    y_data_name = data_name[:-8] + "label.txt"
    print(data_name)
    if y_data_name in label_dict.keys():
        y = np.asarray(label_dict[y_data_name])
    X = np.asarray(X).astype(np.float64)
    result_dict[data_name] = get_result(X, y, classifiers)
    print("\n\n")

print("Training process done")