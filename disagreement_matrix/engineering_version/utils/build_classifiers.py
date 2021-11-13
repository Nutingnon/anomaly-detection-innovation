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
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.pca import PCA
# ensemble
from pyod.models.lscp import LSCP


class Classifiers:
    def __init__(self, base_classifiers_type = 'lofs', random_seed=10, outliers_fraction=0.2):
        if base_classifiers_type == "mixed":
            self.outliers_fraction = outliers_fraction
            self.random_state = np.random.RandomState(random_seed)
            self.detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                             LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                             LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                             LOF(n_neighbors=50), LOF(n_neighbors=55), LOF(n_neighbors=60)]

            self.classifiers = {
                'Histogram-base Outlier Detection (HBOS)': HBOS(
                    contamination=outliers_fraction),

                'Isolation Forest': IForest(contamination=outliers_fraction,
                                            random_state=self.random_state, n_estimators=280),

                # 'K Nearest Neighbors (KNN)': KNN(
                #     contamination=outliers_fraction),

                'Average KNN': KNN(method='mean',
                                   contamination=outliers_fraction),

                'Local Outlier Factor (LOF)':
                    LOF(n_neighbors=30, contamination=outliers_fraction),

                'Principal Component Analysis (PCA)': PCA(
                    contamination=outliers_fraction, random_state=self.random_state),

                "COPOD": COPOD(),

                'Locally Selective Combinatio (LSCP)': LSCP(
                    self.detector_list)
            }
        elif base_classifiers_type == "lofs":
            self.classifiers = dict(
                zip(['LOF_' + str(i) for i in range(5, 61, 5)], [LOF(n_neighbors=x) for x in range(5, 51, 5)])
            )

        else:
            raise KeyError("base_classifiers_type should in ['mixed', 'lofs'].")

    def get_classifiers(self):
        return self.classifiers

    def get_classifiers_length(self):
        return len(self.classifiers)

    def train_helper(self, X, normalizer):
        trained_clfs = []
        clf_scores_dict = dict()
        # Fit the model
        for i, (clf_name, clf) in enumerate(self.classifiers.items()):
            print(clf_name)
            # start_time = time.time()
            clf.fit(X)
            trained_clfs.append((clf_name, clf))
            scores_pred = clf.decision_function(X)
            standard_scores = normalizer().fit_transform(np.reshape(scores_pred, (-1, 1)))
            clf_scores_dict[clf_name] = standard_scores.reshape(-1)
            # end_time = time.time()
            # print("cost", (end_time - start_time) // 60, 'minutes')
            # print('-' * 20)
        return pd.DataFrame.from_dict(clf_scores_dict)











