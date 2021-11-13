'''
for a given matrix X, each column is a normalized y_predict_score, and
each row is one datapoint. and we have several thresholders, if a given
datapoint's disagreement is higher than a certain thresholders, then it
needs mixed up the outlier scores, instead of directly output the main
detector's outlier scores.

INPUT:
- scores_matrix : N x M , N data points, M classifiers
- disagreement series: N x 1
- thresholder methods: K
- factor dict

Output: dict
- Q as keyword,  N x K as value
    Q - main detector selection
    N - data points ensemble scores;
    K - results of different thresholders' with 1 group factor;
'''

import numpy as np
import pandas as pd
from util_functions import UtilFunctionsDisagreement
from threshold_method import Thresholder
from sklearn import metrics


class Constructor:
    def __init__(self, score_matrix, disagreement_series, classifier_names,
                 factor_dict, main_detector_result, main_detector_name):
        self.score_matrix_df = pd.DataFrame(score_matrix, columns=classifier_names)
        self.columns = classifier_names
        self.score_matrix_df["disagreement"] = disagreement_series
        self.score_matrix = score_matrix
        self.disagreement_series = disagreement_series
        self.thresholder_results = Thresholder(disagreement_series).get_thresholder_results(factor_dict)
        self.main_detector_result = main_detector_result
        self.main_detector_name = main_detector_name
        self.score_matrix_df['main_detector'] = main_detector_result

    def predict(self):
        self.result = dict()
        # min
        for threshold in self.thresholder_results.keys():
            self.result['min_'+threshold] = self.score_matrix_df.apply(lambda x: np.min([x[p] for p in self.columns]) if x['disagreement'] >= self.thresholder_results[threshold] else x['main_detector'], axis=1)

        # max
        for threshold in self.thresholder_results.keys():
            self.result['max_'+threshold] = self.score_matrix_df.apply(lambda x: np.max([x[p] for p in self.columns]) if x['disagreement'] >= self.thresholder_results[threshold] else x['main_detector'], axis=1)

        # mean
        for threshold in self.thresholder_results.keys():
            self.result['mean_'+threshold] = self.score_matrix_df.apply(lambda x: np.mean([x[p] for p in self.columns]) if x['disagreement'] >= self.thresholder_results[threshold] else x['main_detector'], axis=1)

        # median
        for threshold in self.thresholder_results.keys():
            self.result['median_'+threshold] = self.score_matrix_df.apply(lambda x: np.median([x[p] for p in self.columns]) if x['disagreement'] >= self.thresholder_results[threshold] else x['main_detector'], axis=1)

        # disagreement_with_softmax
        for order_ in ['o_d_i', 'i_d_o']:
            dis_object = UtilFunctionsDisagreement(order_)
            tmp_res = dis_object.ensemble_disagreement_score(self.score_matrix, len(self.columns))
            self.score_matrix_df['tmp_dis'] = tmp_res
            self.result["disagreement_"+order_] = self.score_matrix_df.apply(lambda x: x['tmp_dis'] if x['disagreement'] >= self.thresholder_results[threshold] else x['main_detector'], axis=1)

            # revert
            self.score_matrix_df = self.score_matrix_df.drop('tmp_dis', axis=1)
        return self.result

    def evaluate(self, y_true):
        assert self.result is not None and len(self.result.keys()) >= 1
        result_df = pd.DataFrame(self.result)
        ensemble_auc = result_df.apply(lambda x: metrics.roc_auc_score(y_true, x))
        base_detector_auc = self.score_matrix_df.loc[:, self.columns].apply(lambda x: metrics.roc_auc_score(y_true, x))
        return ensemble_auc, base_detector_auc





