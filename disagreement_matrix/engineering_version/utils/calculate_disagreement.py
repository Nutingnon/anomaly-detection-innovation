import numpy as np
import pandas as pd

class Disagreement:
    def __init__(self, score_ndarray):
        self.score_matrix = score_ndarray
        self.disagreement_dict = dict()

    def calc_disagreement(self, rsd_matrix=None):
        # standard deviation
        self.disagreement_dict['std'] = np.std(self.score_matrix, axis=1)

        # mad
        median_ = np.median(self.score_matrix, axis=1).reshape((-1,1))
        mad = 1.4826*np.median(np.abs(self.score_matrix - median_), axis=1)
        self.disagreement_dict['mad'] = mad

        columns = ['std', 'mad']

        if rsd_matrix is not None:
            # sum of rsd
            self.disagreement_dict['sum_rsd'] = np.sum(rsd_matrix, axis=1).flatten()

            # std of rsd matrix
            self.disagreement_dict['std_rsd'] = np.std(rsd_matrix, axis=1).flatten()

            # max of rsd matrix
            self.disagreement_dict['max_rsd'] = np.max(rsd_matrix, axis=1).flatten()

            columns.extend(["sum_rsd", 'std_rsd', 'max_rsd'])

        res = pd.DataFrame.from_dict(self.disagreement_dict)
        res = res.loc[:, columns]
        return res

