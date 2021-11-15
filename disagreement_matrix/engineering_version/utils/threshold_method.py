import numpy as np
import scipy as sc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class Thresholder:
    def __init__(self, disagreement_vector):
        self.disagreement_vector = disagreement_vector

    def two_std_thresholder(self, factor=2.5):
        high_limit = np.mean(self.disagreement_vector) + factor * np.std(self.disagreement_vector)
        x_new = self.disagreement_vector[self.disagreement_vector <= high_limit]
        new_std = np.std(x_new)
        new_high_limit = np.mean(x_new) + factor * new_std
        return new_high_limit

    def sd_thresholder(self, factor=2.5):
        high_limit = np.mean(self.disagreement_vector) + factor * np.std(self.disagreement_vector)
        return high_limit

    def mad_thresholder(self, factor=3):
        median_ = np.median(self.disagreement_vector)
        mad = 1.4826*np.median(np.abs(self.disagreement_vector - median_))
        high_limit = factor * mad
        return high_limit

    def iqr_thresholder(self, factor=1.5):
        iqr = np.percentile(self.disagreement_vector, 75) - np.percentile(self.disagreement_vector, 25)
        high_limit = np.percentile(self.disagreement_vector, 75) + factor * iqr
        return high_limit

    def two_std_thresholder(self, factor=2.5):
        high_limit = np.mean(self.disagreement_vector) + factor * np.std(self.disagreement_vector)
        x_new = self.disagreement_vector[self.disagreement_vector <= high_limit]
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

