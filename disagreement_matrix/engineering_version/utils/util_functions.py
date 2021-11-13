import numpy as np
import pandas as pd
import scipy.stats as ss


class UtilFunctionsDisagreement:
    def __init__(self, order_):
        self.order_ = order_

    def softmax(self, scores_array):
        weights = scores_array - max(scores_array)
        s = np.exp(weights).sum()
        weights = np.exp(weights) / s
        #     smooth_scores = weights * scores_array
        return weights

    def get_disagreement_scaled_weight(self, rank_series, ranked_score_matrix):
        dup_a = np.tile(rank_series, (len(rank_series), 1)).astype(int)
        dup_b = np.tile(rank_series, (len(rank_series), 1)).T.astype(int)

        # [[1,2,3], [1,2,3], [1,2,3]]
        idx_m = np.tile(np.arange(len(rank_series)), (len(rank_series), 1))
        tmp_matrix = np.abs(ranked_score_matrix[dup_b, idx_m] - ranked_score_matrix[dup_a, idx_m])
        res = np.maximum(1e-4, tmp_matrix)

        # [[1,1,1],[2,2,2],[3,3,3]]
        idx_n = np.tile(np.arange(len(rank_series)), (len(rank_series), 1)).T

        # sum after divide or divide after sum?
        if self.order_ == "o_d_i":
            weights = np.sum(res[idx_n, idx_m], axis=1) / np.sum(res[idx_m, idx_n], axis=1)
        elif self.order_ == "i_d_o":
            weights = np.sum(res[idx_m, idx_n], axis=1)/np.sum(res[idx_n, idx_m], axis=1)
        #     weights = np.sum(res[idx_n, idx_m] /res[idx_m, idx_n], axis = 1)

        scaled_weights = self.softmax(weights)
        return scaled_weights

    def get_disagreement_matrix(self, rank_series, ranked_score_matrix, which_dis='outer'):
        dup_a = np.tile(rank_series, (len(rank_series), 1)).astype(int)
        dup_b = np.tile(rank_series, (len(rank_series), 1)).T.astype(int)

        # [[1,2,3], [1,2,3], [1,2,3]]
        idx_m = np.tile(np.arange(len(rank_series)), (len(rank_series), 1))
        tmp_matrix = np.abs(ranked_score_matrix[dup_b, idx_m] - ranked_score_matrix[dup_a, idx_m])
        res = np.maximum(1e-4, tmp_matrix)

        # [[1,1,1],[2,2,2],[3,3,3]]
        idx_n = np.tile(np.arange(len(rank_series)), (len(rank_series), 1)).T

        # sum after divide or divide after sum?
        if which_dis == "outer":
            dis_score = np.sum(res[idx_n, idx_m], axis=1)

        elif which_dis == "inner":
            dis_score = np.sum(res[idx_m, idx_n], axis=1)
        return dis_score

    def form_rank_matrix(self, score_matrix):
        sorted_matrix = score_matrix.copy()
        sorted_matrix.sort(axis=0)
        return sorted_matrix

    def ensemble_disagreement_score(self, score_matrix, num_detectors):
        # normalize the score_matrix
        origin_score_matrix = score_matrix.copy()
        score_matrix = score_matrix.to_numpy()
        # get rank matrix
        rank_matrix = np.zeros([len(score_matrix), num_detectors])
        for i in range(num_detectors):
            rank_matrix[:, i] = ss.rankdata(score_matrix[:, i], 'ordinal') - 1
        sorted_score_matrix = self.form_rank_matrix(score_matrix)
        scaled_weights_matrix = np.apply_along_axis(self.get_disagreement_scaled_weight, 1, rank_matrix, sorted_score_matrix)
        assert scaled_weights_matrix.shape == score_matrix.shape
        output = np.sum(score_matrix * scaled_weights_matrix, axis=1)
        # origin_score_matrix["ensemble"] = output
        return output