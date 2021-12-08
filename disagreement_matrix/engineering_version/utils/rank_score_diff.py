import numpy as np
import pandas as pd
import scipy.stats as ss


def softmax(score_row):
    # weights = score_row - max(score_row)
    # s = np.exp(weights).sum()
    # weights = np.exp(weights) / s
    s = np.sum(score_row)
    weights = score_row/s
    return weights


def sort_matrix_by_column(score_matrix):
    sorted_matrix = score_matrix.copy()
    sorted_matrix.sort(axis=0)
    return sorted_matrix


def get_elements_rank_by_column(score_matrix):
    if type(score_matrix) != np.ndarray:
        score_matrix = score_matrix.to_numpy()
    rank_matrix=ss.rankdata(score_matrix, 'ordinal', axis=0) - 1
    return rank_matrix


def get_rank_score_diff_in_one_row(rank_series, ranked_score_matrix):
    dup_a = np.tile(rank_series, (len(rank_series), 1)).astype(int)
    dup_b = np.tile(rank_series, (len(rank_series), 1)).T.astype(int)
    # [[1,2,3], [1,2,3], [1,2,3]]
    idx_m = np.tile(np.arange(len(rank_series)), (len(rank_series), 1))
    tmp_matrix = np.abs(ranked_score_matrix[dup_b, idx_m] - ranked_score_matrix[dup_a, idx_m])
    res = np.maximum(1e-5, tmp_matrix)
    # res.reshape(-1)
    res.reshape(len(rank_series)**2, -1)
    return res.reshape(len(rank_series)**2, -1)


# output one row property. input 1xD, output: 1xD
def get_rank_score_diff_datapoint_property(one_dim_dis_one_row):
    n_dim = int(np.sqrt(len(one_dim_dis_one_row)))
    dis_matrix_one_row = one_dim_dis_one_row.reshape((n_dim, n_dim))
    # [[0,1,2],[0,1,2],[0,1,2]]
    idx_m = np.tile(np.arange(n_dim), (n_dim, 1))
    # [[0,0,0],[1,1,1],[2,2,2]]
    idx_n = idx_m.T
    # 1 x D
    outer_score_sum = np.maximum(np.sum(dis_matrix_one_row[idx_n, idx_m], axis=1).reshape(-1), 1e-5)
    # 1 x D
    inner_score_sum = np.maximum(np.sum(dis_matrix_one_row[idx_m, idx_n], axis=1).reshape(-1), 1e-5)
    # 1 X D
    o_d_i = softmax(outer_score_sum/inner_score_sum)
    # 1 X D
    i_d_o = softmax(inner_score_sum/outer_score_sum)
    # res_dict = {"outer": outer_score_sum, "inner": inner_score_sum, 'o_d_i': o_d_i, 'i_d_o': i_d_o}
    res_array = [outer_score_sum, inner_score_sum, o_d_i, i_d_o]
    return res_array


def translate_property(key, res_array):
    if key == 'outer':
        return res_array[:, 0, :]
    elif key == "inner":
        return res_array[:, 1, :]
    elif key == "o_d_i":
        return res_array[:, 2, :]
    elif key == "i_d_o":
        return res_array[:, 3, :]
    else:
        raise KeyError("Unknown Key: ", key)


class RankScoreDiff:
    # scores_array is an nd_array
    def __init__(self, scores_array):
        self.rsdm_2d = np.empty_like(scores_array)
        self.rank_matrix = None
        self.sorted_matrix = None
        self.scores_array = scores_array
        self.outer_score_matrix = None
        self.inner_score_matrix = None
        self.o_d_i = None
        self.i_d_o = None
        self.sum_self = None
        self.sum_self_inverse = None

    def generate_rank_score_diff_property(self):
        sorted_matrix = sort_matrix_by_column(self.scores_array)
        self.sorted_matrix = sorted_matrix
        rank_records = get_elements_rank_by_column(self.scores_array)
        self.rank_matrix = rank_records
        res_dis_vector = np.apply_along_axis(get_rank_score_diff_in_one_row, 1, rank_records, sorted_matrix)
        self.rsdm_2d = res_dis_vector
        property_array = np.apply_along_axis(get_rank_score_diff_datapoint_property, 1, self.rsdm_2d)
        self.outer_score_matrix = translate_property('outer', property_array)
        self.inner_score_matrix = translate_property('inner', property_array)
        self.o_d_i = translate_property('o_d_i', property_array).reshape((sorted_matrix.shape[0], -1))
        self.i_d_o = translate_property('i_d_o', property_array).reshape((sorted_matrix.shape[0], -1))
        self.sum_self = self.outer_score_matrix + self.inner_score_matrix
        sum_self_inverse = np.reciprocal(self.sum_self)
        sum_self_inverse = sum_self_inverse/np.sum(sum_self_inverse)
        self.sum_self = np.reshape(self.sum_self/np.sum(self.sum_self),(sorted_matrix.shape[0], -1))
        self.sum_self_inverse = np.reshape(sum_self_inverse, (sorted_matrix.shape[0], -1))
        return self

    def get_rsd_matrix_2d(self):
        return self.rsdm_2d


