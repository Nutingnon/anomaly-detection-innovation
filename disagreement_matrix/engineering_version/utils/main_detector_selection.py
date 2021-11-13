import numpy as np
import sys
import os
import pandas as pd
from util_functions import UtilFunctionsDisagreement
from threshold_method import Thresholder
import scipy.stats as ss

class MainDetectorSelection:
    def __init__(self, predict_score_df):
        self.X = predict_score_df
        self.disagreement_records = dict()
        self.util_func = UtilFunctionsDisagreement

    def computer_disagreement(self):
        # std
        self.disagreement_records['std'] = self.X.apply(lambda x: np.std(x), axis = 0)

        # mad
        self.disagreement_records['mad'] = self.X.apply(lambda x: Thresholder(x).mad_thresholder(1), axis=0)

        # disagreement
        score_m_np = self.X.to_numpy()
        rank_matrix = np.zeros_like(score_m_np)
        for i in range(rank_matrix.shape[1]):
            rank_matrix[:, i] = ss.rankdata(score_m_np[:, i], 'ordinal') - 1
        sorted_score_matrix = self.util_func.form_rank_matrix(score_m_np)

        ## outer disagreement
        self.disagreement_records['outer_disagree_matrix'] = np.apply_along_axis(self.util_func.get_disagreement_matrix, 1, rank_matrix,
                                                    sorted_score_matrix, 'outer')



        # inner disagreement
        self.disagreement_records['inner_disagree_matrix'] = np.apply_along_axis(self.util_func.get_disagreement_matrix, 1, rank_matrix,
                                                    sorted_score_matrix, 'inner')


        # random
        self.disagreement_records['random'] = pd.DataFrame(np.random.random((self.X.shape[0], self.X.shape[1])), columns=self.X.columns)

    def get_main_detector_results(self):
        # min global std

        # max global std

        # min outer disagreement

        # min inner disagreement

        # max outer disagreement

        # max inner disagreement

        # random
