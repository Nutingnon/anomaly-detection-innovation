import numpy as np
import scipy as sc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def sd_thresholder(vector, factor=2.5):
    high_limit = np.mean(vector) + factor * np.std(vector)
    return high_limit


def mad_thresholder(vector, factor=3):
    median_ = np.median(vector)
    mad = 1.4826 * np.median(np.abs(vector - median_))
    high_limit = factor * mad
    return high_limit


def iqr_thresholder(vector, factor=1.5):
    iqr = np.percentile(vector, 75) - np.percentile(vector, 25)
    high_limit = np.percentile(vector, 75) + factor * iqr
    return high_limit


def two_std_thresholder(vector, factor=2.5):
    high_limit = np.mean(vector) + factor * np.std(vector)
    x_new = vector[vector <= high_limit]
    new_std = np.std(x_new)
    new_high_limit = np.mean(x_new) + factor * new_std
    return new_high_limit


# return [4, N, 5]
def get_thresholder_results(vector, factor_dict=None):
    threshold_dict = dict()
    # sorted key is aligned with the following sequence. [2std, iqr, mad, std]
    if factor_dict is not None:
        threshold_dict['2std'] = two_std_thresholder(vector, factor_dict['2std'])
        threshold_dict['iqr'] = iqr_thresholder(vector, factor_dict['iqr'])
        threshold_dict['mad'] = mad_thresholder(vector, factor_dict['mad'])
        threshold_dict['std'] = sd_thresholder(vector, factor_dict['std'])
    else:
        threshold_dict['2std'] = [two_std_thresholder(vector)]
        threshold_dict['iqr'] = [iqr_thresholder(vector)]
        threshold_dict['mad'] = [mad_thresholder(vector)]
        threshold_dict['std'] = [sd_thresholder(vector)]

    sorted_list = sorted(threshold_dict.items(), key=lambda x: x[0])
    sorted_threshold = [x[1] for x in sorted_list]
    s = np.asarray(sorted_threshold).reshape((4, 1))
    return np.less_equal(vector, s)
