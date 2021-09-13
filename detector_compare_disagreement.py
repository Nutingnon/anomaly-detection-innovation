from __future__ import division,print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP # ensemble
import multiprocessing as mp
from IPython.display import display
from thresholders import *


data_path = "/Users/kadima/experiment_any/anomaly-detection/datasets/"
def getData(fileName):
    # check fileName:
    files = [x for x in os.listdir(data_path) if x.endswith(".mat")]
    mat = scipy.io.loadmat(data_path+fileName)
    X = mat["X"]
    y = mat["y"]
    return X, y

def get_score_matrix(X, num_detectors):
    return np.zeros([X.shape[0], num_detectors])

def get_perform_matrix(num_thresholders, num_detectors):
    return np.zeros((num_thresholders, num_detectors))


random_state = np.random.RandomState(10)
outliers_fraction = 0.4
# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                       contamination=outliers_fraction,
                       random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state, n_estimators = 280),
    'K Nearest Neighbors (KNN)': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    'Minimum Covariance Determinant (MCD)': MCD(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Principal Component Analysis (PCA)': PCA(
        contamination=outliers_fraction, random_state=random_state),
    'Locally Selective Combination (LSCP)': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state)
}

names = []
# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    names.append(clf)
#     print('Model', i + 1, clf)

# Fit the models with the generated data and 
# compare model performances
def get_result(X, y, classifiers):
    threshold_records = list()
    # create matrix to store the performance
    score_matrix = get_score_matrix(X, len(classifiers.keys()))
    perform_table = get_perform_matrix(5, len(classifiers.keys()) )

    np.random.seed(5)
    clfs = []
    # Fit the model
    for i, (clf_name, clf) in enumerate(classifiers.items()):
#         print(i + 1, 'fitting', clf_name)
        # fit the data and tag outliers
        clf.fit(X)
        clfs.append(clf)
        scores_pred = clf.decision_function(X)
        score_matrix[:, i] = scores_pred
        

    for i, thresholder in enumerate([sd_thresholder, mad_thresholder, 
                                     iqr_thresholder]):
        kk = []
        for j in range(score_matrix.shape[1]):
            _,perform_table[i,j],b = thresholder(score_matrix[:,j], y)
            kk.append(b)
            
        threshold_records.append(kk)
            
    for i in range(score_matrix.shape[1]):
        perform_table[-2, i] = f1_score(y,clfs[i].predict(X))
        
    perform_table[-1,:], d_threshold, disagreement_record = disagreement_vector(score_matrix, len(classifiers), y, False)
    threshold_records.append(d_threshold)
    
    return (pd.DataFrame(perform_table, columns = names, 
            index = ["sd",'mad','iqr','default','disagreement']), 
            threshold_records, disagreement_record, score_matrix)

def run_helper(datasets):
    X,y = getData(datasets)
    X = X.astype(np.float64)
    return get_result(X, y, classifiers)

if __name__ == "__main__":
    mp.freeze_support()
    result_dict = dict()
    bug_datasets = ["letter.mat","speech.mat","cardio.mat"]
    with mp.Pool(8) as pool:
        for datasets in os.listdir("/Users/kadima/experiment_any/anomaly-detection/datasets/"):
            if datasets in bug_datasets:
                continue
            print(datasets)
            result_dict[datasets] = pool.apply_async(run_helper, (datasets,))\
            
        pool.close()
        pool.join()
        for datasets in os.listdir("/Users/kadima/experiment_any/anomaly-detection/datasets/"):
            if datasets in bug_datasets:
                continue
            else:
                result_dict[datasets] = result_dict[datasets].get()


    display(result_dict["lympho.mat"][0])
    