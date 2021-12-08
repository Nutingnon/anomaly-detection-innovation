'''
for a given matrix X, each column is a normalized y_predict_score, and
each row is one datapoint. and we have several thresholders, if a given
datapoint's disagreement is higher than a certain thresholders, then it
needs mixed up the outlier scores, instead of directly output the main
detector's outlier scores.

INPUT:
- score_df : N x M , N data points, M classifiers
- disagreement_results: 4 x N x 5,
    4: [2std, iqr, mad, std],
    5: ['std', 'mad', "sum_rsd", 'std_rsd', 'max_rsd']
- rsd_obj
- main detector name

OUTPUT:
- combination results, which contains 6 x 4 x 5 = 120 columns.

For a file. then get average over them get final 1 result.
'''


import numpy as np
import pandas as pd
from . import threshold_method
from sklearn import metrics


class Constructor:
    def __init__(self, score_df, main_detector_name, rsd_obj, disagreement_result_ndarray):

        self.score_df = score_df
        self.columns = score_df.columns
        self.main_detector_name = main_detector_name
        self.score_matrix = score_df.to_numpy()
        self.disagreement_ndarray_boolean = disagreement_result_ndarray
        self.rsd_obj = rsd_obj
        ensemble_res = dict()
        # max
        ensemble_res['max']=np.max(self.score_matrix, axis=1)
        # min
        ensemble_res['min']=np.min(self.score_matrix, axis=1)
        # mean
        ensemble_res['mean']=np.mean(self.score_matrix, axis=1)
        # median
        ensemble_res['median'] = np.mean(self.score_matrix, axis=1)

        # softmax with ido
        ensemble_res['sido'] = np.sum(self.score_matrix * self.rsd_obj.i_d_o, axis=1)

        # softmax with odi
        ensemble_res['sodi'] = np.sum(self.score_matrix * self.rsd_obj.o_d_i, axis=1)

        # pure sum
        ensemble_res['sum_self'] = np.sum(self.score_matrix * np.reshape(self.rsd_obj.sum_self,(-1, self.score_matrix.shape[1])), axis=1)

        # sum inverse
        ensemble_res['sum_self_inverse'] = np.sum(self.score_matrix * np.reshape(self.rsd_obj.sum_self_inverse,(-1, self.score_matrix.shape[1])), axis=1)


        self.ensemble_result_df = pd.DataFrame.from_dict(ensemble_res)
        self.ensemble_result_df = self.ensemble_result_df.loc[:, ['max','min','mean','median','sido','sodi','sum_self', 'sum_self_inverse']]
        self.ensemble_result_ndarray = self.ensemble_result_df.to_numpy()
        self.main_detector_array = self.score_df[self.main_detector_name].to_numpy().reshape((-1, 1))
        self.result_compat = None
        self.result_flatten_df = None

    def predict(self, flatten_columns):
        # main detector result [N x 1]
        # thresh_dis_res [4, N, 5] boolean value
        # construct res [N, 6]
        # first, we need to assign main detector result
        fill_main_detector_score = self.disagreement_ndarray_boolean * self.main_detector_array
        tmp_results = []
        for col in range(self.ensemble_result_ndarray.shape[1]):
            ensem_res_tmp = self.ensemble_result_ndarray[:, [col]]
            tmp_res = np.where(fill_main_detector_score>0, fill_main_detector_score, ensem_res_tmp)
            tmp_results.append(tmp_res)
        self.result_compat = tmp_results
        result_flatten = np.concatenate(self.result_compat)

        result_flatten = np.vstack([result_flatten[:, i, :].reshape(-1)
                                    for i in range(self.ensemble_result_df.shape[0])])

        self.result_flatten_df = pd.DataFrame(result_flatten, columns=flatten_columns)



    def evaluate(self, y_true):
        assert self.result is not None and len(self.result.keys()) >= 1
        result_df = pd.DataFrame(self.result)
        ensemble_auc = result_df.apply(lambda x: metrics.roc_auc_score(y_true, x))
        base_detector_auc = self.score_df.loc[:, self.columns].apply(lambda x: metrics.roc_auc_score(y_true, x))
        return ensemble_auc, base_detector_auc





