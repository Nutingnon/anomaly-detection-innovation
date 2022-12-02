import pyod
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import os
from pyod.models.copod import COPOD
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.utils.utility import standardizer
from pyod.models.lscp import LSCP
from sklearn.preprocessing import RobustScaler
import multiprocessing as mp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


df = pd.read_excel("C:/Users/froke/jupyter_notebook_files/tain_data.xlsx")
X = df.iloc[:,1:-1]
y = df.iloc[:, -1]


# initialized a group of outlier detectors for acceleration
detector_list = [LOF(n_neighbors=30),
                 COPOD(),
                 IForest(n_estimators=100),
                 IForest(n_estimators=200),
                 HBOS(),
                 KNN()]


def get_res_on_data(clf, X):
    clf.fit(X)
    scores = clf.decision_function(X)
    y_pred = clf.predict(X)
    return clf, scores, y_pred

clfs = []
scores_list = []
cnt = 1
precision_list = []
recall_list = []
c_matrix = []

if __name__ == "__main__":
    a_list = []
    with mp.Pool(7)as pool:
        for detector in detector_list:
            a_list.append((pool.apply_async(get_res_on_data, (detector, X))))
        pool.close()
        pool.join()

    for element in a_list:
        clf, score, y_pred = element.get(timeout=10)
        clfs.append(clf)
        scores_list.append(score)
        precision_list.append(precision_score(y, y_pred, average='binary'))
        recall_list.append(recall_score(y, y_pred))
        c_matrix.append(confusion_matrix(y, y_pred))

    for i in range(6):
        print(i, 'auc', roc_auc_score(y, round(scores_list[i]),3))
        print(i, 'precision', round(precision_list[i],3))
        print(i, 'recall', round(recall_list[i]))
        print(i, 'confusion_matrix', c_matrix[i])
        print()
        print("=="*20)
        print()