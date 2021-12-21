import numpy as np
import pandas as pd
import scipy.stats as ss
import sys
import os
import time
from utils.data_loader import DataLoader
from utils.build_classifiers import Classifiers
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from IPython.display import display
from utils.calculate_disagreement import Disagreement
from utils.rank_score_diff import RankScoreDiff
from utils.threshold_method import *
from utils.construct_method import Constructor
from utils.other_functions import *
from sklearn import metrics
"""
This is DisagrEement EnseMble (DEEM) API, which get several things as input

@ INPUT:
    - scores_detectors: numpy.ndarray with (N, D). Each Column is a scores of a certain Detector. By default, 
                        the larger the score , the instance has higher probability to be an outlier.
                        
    - normalizer :  a normalizer you choose to normalize your scores.
                        
    - run_mode:     a keyword in ['based_on_main_detector', 'auto']. if the keyword='auto', then it change all
                    the instance scores based on the rank score diff weights.

    - label (Optional): numpy.ndarray(N, 1). if y = 1, then it is an outlier. This is optional parameter which help
                        you to evaluate auc or automatically find the main detector.
    
    - main_detector_index:  if you want to mannualy assigned the main_detector, then give the index. please note that
                            this parameter only works when run_mode = 'based_on_main_detector'.
                            
@ OUTPUT:
    - prediction_scores: (N x M), N is the number of instances. There are M ways to combine different strategy. 
    - AUC: (M x 1) if labels are given, it can give the AUC of each combination of prediction scores
"""


class DEEM:
    def __init__(self, score_detectors, normalizer, run_mode='auto', label=None, main_detector_index=None):
        ensemble_sequence = ['max', 'min', 'mean', 'median', 'sido', 'sodi', 'sum_self', 'sum_self_inverse']
        disagreement_sequence = ['std', 'mad', 'sum_rsd', 'std_rsd', 'max_rsd']
        threshold_sequence = ['2std', 'iqr', 'mad', 'std']
        self.main_detector = main_detector_index
        self.score_detectors = score_detectors
        # normalize the scores
        score_detectors = normalizer.fit_transform(score_detectors)
        self.flatten_predict_score_df = None

        a_col = []
        for en in ensemble_sequence:
            for thr in threshold_sequence:
                for dis in disagreement_sequence:
                    a_col.append("ensem_" + en + ":" + "thr_" + thr + ":" + "dis_" + dis)

        if run_mode == 'auto':
            main_detector_index = 0
            rsd_obj = get_rsd_obj(self.score_detectors)
            # dataframe
            disagreement_res = get_disagreement(self.score_detectors, rsd_obj)
            # print("\tDone in calculate disagreement")
            # matrix: [2std, iqr, mad, std]
            thresholders = get_thresholders(disagreement_res.to_numpy(), zeros_like=True)
            # print("\tDone in calculate thresholding")
            self.flatten_predict_score_df = get_construct_methods(self.score_detectors, main_detector_index, rsd_obj,
                                                                    thresholders, a_col)

        elif run_mode == "based_on_main_detector":
            if main_detector_index is not None:
                rsd_obj = get_rsd_obj(score_detectors)
                # dataframe
                disagreement_res = get_disagreement(score_detectors, rsd_obj)
                # print("\tDone in calculate disagreement")
                # matrix: [2std, iqr, mad, std]
                thresholders = get_thresholders(disagreement_res.to_numpy(), zeros_like=False)
                # print("\tDone in calculate thresholding")
                self.flatten_predict_score_df = get_construct_methods(score_detectors, main_detector_index, rsd_obj,
                                                                        thresholders, a_col)
            else:
                if label is not None:
                    # find the best auc's detector
                    performance = np.apply_along_axis(metrics.roc_auc_score, 0, label, self.score_detectors)
                    main_detector_index = np.argmax(performance)
                    rsd_obj = get_rsd_obj(score_detectors)
                    # dataframe
                    disagreement_res = get_disagreement(score_detectors, rsd_obj)
                    # print("\tDone in calculate disagreement")
                    # matrix: [2std, iqr, mad, std]
                    thresholders = get_thresholders(disagreement_res.to_numpy(), zeros_like=False)
                    # print("\tDone in calculate thresholding")
                    self.flatten_predict_score_df = get_construct_methods(score_detectors, main_detector_index, rsd_obj,
                                                                            thresholders, a_col)


    def get_ensemble_score(self):
        return self.flatten_predict_score_df
