import numpy as np
import sys
import os
import pandas as pd
from rank_score_diff import RankScoreDiff
from threshold_method import Thresholder
import scipy.stats as ss

class MainDetectorSelection:
    def __init__(self, predict_score_df):
        self.X = predict_score_df
        self.disagreement_records = dict()
        self.util_func = RankScoreDiff

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
        sorted_score_matrix = self.util_func.sort_matrix_by_column(score_m_np)

        ## outer disagreement
        self.disagreement_records['outer_disagree_matrix'] = np.apply_along_axis(self.util_func.get_disagreement_matrix, 1, rank_matrix,
                                                    sorted_score_matrix, 'outer')
        assert self.disagreement_records['outer_disagree_matrix'].shape[0] == self.X.shape[0]
        assert self.disagreement_records['outer_disagree_matrix'].shape[1] == self.X.shape[1]


        # inner disagreement
        self.disagreement_records['inner_disagree_matrix'] = np.apply_along_axis(self.util_func.get_disagreement_matrix, 1, rank_matrix,
                                                    sorted_score_matrix, 'inner')
        assert self.disagreement_records['inner_disagree_matrix'].shape[0] == self.X.shape[0]
        assert self.disagreement_records['inner_disagree_matrix'].shape[1] == self.X.shape[1]

        # random
        self.disagreement_records['random'] = pd.DataFrame(np.random.random((self.X.shape[0], self.X.shape[1])), columns=self.X.columns)

    def get_main_detector_results(self, random_seed):
        np.random.seed(random_seed)
        result = dict()
        # min global std
        main_detector = self.disagreement_records['std'].columns[np.argmin(self.disagreement_records['std'].to_numpy())]
        result['min_std'] = self.X.loc[:, main_detector].to_numpy()

        # max global std
        main_detector = self.disagreement_records['std'].columns[np.argmax(self.disagreement_records['std'].to_numpy())]
        result['max_std'] = self.X.loc[:, main_detector].to_numpy()

        # min global mad
        main_detector = self.disagreement_records['mad'].columns[np.argmin(self.disagreement_records['mad'].to_numpy())]
        result['min_mad'] = self.X.loc[:, main_detector].to_numpy()

        # max global mad
        main_detector = self.disagreement_records['mad'].columns[np.argmax(self.disagreement_records['mad'].to_numpy())]
        result['max_mad'] = self.X.loc[:, main_detector].to_numpy()

        # min global outer disagreement
        argmin_idx = np.argmin(np.sum(self.disagreement_records["outer_disagree_matrix"], axis=0))
        result['min_outer_global_disagreement'] = self.X.iloc[:, argmin_idx].to_numpy()

        # max global outer disagreement
        argmax_idx = np.argmax(np.sum(self.disagreement_records["outer_disagree_matrix"], axis=0))
        result['max_outer_global_disagreement'] = self.X.iloc[:, argmax_idx].to_numpy()

        # min global inner disagreement
        argmin_idx = np.argmin(np.sum(self.disagreement_records["inner_disagree_matrix"], axis=0))
        result['min_inner_global_disagreement'] = self.X.iloc[:, argmin_idx].to_numpy()

        # max global inner disagreement
        argmax_idx = np.argmax(np.sum(self.disagreement_records["inner_disagree_matrix"], axis=0))
        result['max_inner_global_disagreement'] = self.X.iloc[:, argmax_idx].to_numpy()

        # max local outer disagreement
        max_idx = np.argmax(self.disagreement_records["outer_disagree_matrix"], 1)
        result["max_outer_local_disagreement"] = self.X.iloc[np.arange(len(self.X)), max_idx].to_numpy()

        # min local outer disagreement
        min_idx = np.argmin(self.disagreement_records["outer_disagree_matrix"], 1)
        result["min_outer_local_disagreement"] = self.X.iloc[np.arange(len(self.X)), min_idx].to_numpy()

        # min local inner disagreement
        min_idx = np.argmin(self.disagreement_records["inner_disagree_matrix"], 1)
        result["min_inner_local_disagreement"] = self.X.iloc[np.arange(len(self.X)), min_idx].to_numpy()

        # max local inner disagreement
        max_idx = np.argmax(self.disagreement_records["inner_disagree_matrix"], 1)
        result["max_inner_local_disagreement"] = self.X.iloc[np.arange(len(self.X)), max_idx].to_numpy()

        # global_random
        result['global_random'] = self.X.iloc[:, np.random.randint(0, self.X.shape[1])]

        # local_random
        result['local_random'] = self.X.iloc[np.arange(self.X.shape[0]), np.random.choice(np.arange(self.X.shape[1]), self.X.shape[0])]

        return result
