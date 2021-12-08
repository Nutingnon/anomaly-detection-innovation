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
from sklearn import metrics

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
    def __init__(self, base_classifiers_type='lofs', random_seed=22, outliers_fraction=0.05):
        if base_classifiers_type == "mixed":
            self.outliers_fraction = outliers_fraction
            self.random_state = np.random.RandomState(random_seed)

            # self.detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
            #                  LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
            #                  LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
            #                  LOF(n_neighbors=50), LOF(n_neighbors=55), LOF(n_neighbors=60)]
            self.detector_list = [
                HBOS(contamination=self.outliers_fraction),
                IForest(contamination=self.outliers_fraction, random_state=self.random_state, n_estimators=280),
                KNN(contamination=outliers_fraction),
                KNN(method='mean', contamination=outliers_fraction),
                LOF(n_neighbors=30, contamination=outliers_fraction),
                PCA(contamination=outliers_fraction, random_state=self.random_state),
                COPOD()
            ]
            self.score_df = pd.DataFrame()
            self.main_detector = None
            self.performance = None
            self.y = None
            self.classifiers = {
                'Histogram-base Outlier Detection (HBOS)': HBOS(
                    contamination=outliers_fraction),

                'Isolation Forest': IForest(contamination=outliers_fraction,
                                            random_state=self.random_state, n_estimators=280),

                'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),

                'Average KNN': KNN(method='mean',contamination=outliers_fraction),

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
                zip(['LOF_' + str(i) for i in range(10, 80, 10)], [LOF(n_neighbors=x) for x in range(10, 80, 10)])
            )

        else:
            raise KeyError("base_classifiers_type should in ['mixed', 'lofs'].")

    def get_classifiers(self):
        return self.classifiers

    def get_classifiers_length(self):
        return len(self.classifiers)

    def train_helper(self, X, y, normalizer, fname):
        trained_classifiers = []
        clf_scores_dict = dict()
        clf_names = []
        # Fit the model
        for i, (clf_name, clf) in enumerate(self.classifiers.items()):
            clf.fit(X)
            trained_classifiers.append((clf_name, clf))
            scores_pred = clf.decision_function(X)
            standard_scores = normalizer().fit_transform(np.reshape(scores_pred, (-1, 1)))
            clf_scores_dict[clf_name] = standard_scores.reshape(-1)
            clf_names.append(clf_name)
        self.score_df = pd.DataFrame.from_dict(clf_scores_dict)
        # print(clf_names)

        # calculate AUC score for each base detector to
        performance = self.score_df.apply(lambda x: metrics.roc_auc_score(y, x), axis=0)
        # output performance
        pd.DataFrame(performance).to_excel("/Users/kadima/experiment_any/anomaly-detection/disagreement_matrix/engineering_version/auc_results_sub/" +fname + ".xlsx")
        self.performance = performance
        best_performance = self.score_df.columns[np.argmax(performance)]
        self.main_detector = best_performance
        self.y = y
        return self

    def get_auc_on_base_classifier(self, X, y, output_path1, fname):
        self.performance.to_excel(output_path1, sheet_name=fname)











