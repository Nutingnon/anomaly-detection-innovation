import numpy as np
import scipy as sc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class Thresholder:
    def __init__(self, x):
        self.x = x

    def sd_thresholder(self, factor=2.5):
        high_limit = np.mean(self.x) + factor * np.std(self.x)
        return high_limit

    def mad_thresholder(self, factor=3):
        median_ = np.median(self.x)
        mad = 1.4826*np.median(np.abs(self.x - median_))
        high_limit = factor * mad
        return high_limit

    def iqr_thresholder(self, factor = 1.5):
        iqr = np.percentile(self.x, 75) - np.percentile(self.x, 25)
        high_limit = np.percentile(self.x, 75) + factor * iqr
        return high_limit

    def two_std_thresholder(self, factor = 2.5):
        high_limit = np.mean(self.x) + factor * np.std(self.x)
        x_new = self.x[self.x<=high_limit]
        new_std = np.std(x_new)
        new_high_limit = np.mean(x_new) + factor * new_std
        return new_high_limit

    def get_thresholder_results(self, factor_dict):
        threshold_dict = dict()
        threshold_dict['std'] = self.sd_thresholder(factor_dict['std'])
        threshold_dict['2std'] = self.two_std_thresholder(factor_dict['2std'])
        threshold_dict['iqr'] = self.iqr_thresholder(factor_dict['iqr'])
        threshold_dict['mad'] = self.mad_thresholder(factor_dict['mad'])
        return threshold_dict

