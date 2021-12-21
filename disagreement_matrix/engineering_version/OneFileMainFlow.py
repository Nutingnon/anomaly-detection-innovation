import numpy as np
import pandas as pd
import scipy.stats as ss
import sys
import os
import time
# import utils
## read data
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
from sklearn import metrics


# Main idea: 一个外套走完所有数据集
# read data
def load_data():
    dataloader = DataLoader()
    train_X, train_y = dataloader.get_train_data()
    return train_X, train_y


def filter_big_data(train_X, train_y, MB):
    a = []
    b = []
    c = []
    for fname, data in train_X.items():
        df_tmp = pd.DataFrame(data)
        a.append(fname)
        b.append(round(df_tmp.memory_usage(deep=True).sum() * 1e-6, 4))
        c.append(str(round(sum(train_y[fname]) / len(train_y[fname]) * 100, 4)) + "%")
    df_discription = pd.DataFrame({"FileName": a, "Memory_in_MB": b, "OutliersRate": c})
    small_sets = set(df_discription.FileName[df_discription.Memory_in_MB <= MB])
    new_train_X = dict()
    new_train_y = dict()
    for fname, data in train_X.items():
        if fname in small_sets:
            new_train_X[fname] = train_X[fname]
            new_train_y[fname] = train_y[fname]
    return new_train_X, new_train_y


def train_classifiers():
    x, y = load_data()
    training_results = dict()
    train_X, train_y = filter_big_data(x, y, 60)
    normalizer = RobustScaler # RobustScaler
    score_dfs = dict()
    clf_initializer = Classifiers('mixed')
    main_detectors = dict()
    performances = dict()
    mp.freeze_support()
    with mp.Pool(10) as pool:
        # files, data
        # data is an numpy ndarray
        for fname, data in train_X.items():
            # base classifiers
            training_results[fname] = pool.apply_async(clf_initializer.train_helper, (data, train_y[fname], normalizer, fname))
        pool.close()
        pool.join()
        for key in training_results:
            training_results[key] = training_results[key].get(timeout=5)
    return training_results


def get_disagreement(score_df, rsdiff_obj):
    score_ndarray = score_df.to_numpy()
    dis_obj = Disagreement(score_ndarray)
    disagreement_df = dis_obj.calc_disagreement(rsdiff_obj.get_rsd_matrix_2d())
    return disagreement_df


def get_rsd_obj(score_nd_array):
    tmp_var = RankScoreDiff(score_nd_array)
    rsdiff_obj = tmp_var.generate_rank_score_diff_property()
    return rsdiff_obj


def get_thresholders(disagreement_nd_array,zeros_like=False):
    s = np.apply_along_axis(get_thresholder_results, 0, disagreement_nd_array, zeros_like)
    return s


def get_construct_methods(score_df, main_detector_name, rsd_obj, disagreement_result_ndarray, flatten_columns):
    constructor = Constructor(score_df, main_detector_name, rsd_obj, disagreement_result_ndarray)
    constructor.predict(flatten_columns)
    return constructor.result_flatten_df



if __name__ == "__main__":
    rank_score_diff_obj_collection_dict = dict()
    disagreement_collection_dict = dict()
    thresholder_is_exceed_matrix_collection = dict()

    ############################################################################################
    '''
    Step 1. Calculate Scores from Base Detectors for each file.
    base_results is a dictionary with filename as key, and its value is a classifier object
    which has 3 important components: 
        1. score_df: dataframe
        2. performance: AUC DataFrame
        3. String, Main detector's name
    '''
    # base results is a dictionary
    # its key is file name and its value is a DataFrame with results from scores.
    base_results = train_classifiers()
    flatten_predict_score_df = dict()
    ensemble_sequence = ['max', 'min', 'mean', 'median', 'sido', 'sodi','sum_self','sum_self_inverse']
    disagreement_sequence = ['std', 'mad', 'sum_rsd', 'std_rsd', 'max_rsd']
    threshold_sequence = ['2std', 'iqr', 'mad', 'std']
    a_col = []
    for en in ensemble_sequence:
        for thr in threshold_sequence:
            for dis in disagreement_sequence:
                a_col.append("ensem_" + en + ":" + "thr_" + thr + ":" + "dis_" + dis)
    print("Base Classifier Trainning Completed, head to step2")
    ############################################################################################
    '''
    Step2. Calculate properties for each file 
    -   Disagreement matrix for each file. Disagreement has 5 calculation methods.
        So, for each file, it is a DataFrame with Nx5 shape;
    -   RSD_OBJ. Contains rank score difference object;
    -   Thresholder. Contains 4 thresholder.
    '''
    auc_res = dict()
    main_detector_dict = dict()
    for fname in base_results.keys():
        print("Processing ", fname)
        # dataframe
        score_df = base_results[fname].score_df
        main_detector = base_results[fname].main_detector
        main_detector_dict[fname] = main_detector
        best_auc = base_results[fname].performance[main_detector]
        rsd_obj = get_rsd_obj(score_df.to_numpy())
        rank_score_diff_obj_collection_dict[fname] = rsd_obj
        print("\tDone in Calculate RSD Matrix")


        # dataframe
        disagreement_collection_dict[fname] = get_disagreement(score_df, rsd_obj)
        print("\tDone in calculate disagreement")

        # matrix: [2std, iqr, mad, std]
        thresholder_is_exceed_matrix_collection[fname] = get_thresholders(disagreement_collection_dict[fname].to_numpy(), zeros_like=False)
        # write out
        pd.DataFrame(thresholder_is_exceed_matrix_collection[fname][3, :, 4]).to_excel("/Users/kadima/experiment_any/anomaly-detection/disagreement_matrix/engineering_version/file_disagreement_records/"+fname+"_threshold.xlsx")



        print("\tDone in calculate thresholding")

        flatten_predict_score_df[fname] = get_construct_methods(score_df, main_detector, rsd_obj, thresholder_is_exceed_matrix_collection[fname],
                                                                a_col)

        # write out rsj matrix
        pd.DataFrame(rank_score_diff_obj_collection_dict[fname].o_d_i).to_excel("/Users/kadima/experiment_any/anomaly-detection/disagreement_matrix/engineering_version/rsj_matrix_records/"+fname+"_odi.xlsx")
        pd.DataFrame(rank_score_diff_obj_collection_dict[fname].i_d_o).to_excel("/Users/kadima/experiment_any/anomaly-detection/disagreement_matrix/engineering_version/rsj_matrix_records/"+fname + "_ido.xlsx")
        pd.DataFrame(rank_score_diff_obj_collection_dict[fname].sum_self).to_excel("/Users/kadima/experiment_any/anomaly-detection/disagreement_matrix/engineering_version/rsj_matrix_records/"+fname + "_self_sum.xlsx")
        print("\tDone in calculate construct")

        auc_res[fname] = flatten_predict_score_df[fname].apply(lambda x: metrics.roc_auc_score(base_results[fname].y, x), axis=0)
        print("\tDone in calculate all auc")
        auc_res[fname] = auc_res[fname].append(pd.Series([best_auc], index=["best_base_auc"]))

        # write out the auc_res
        auc_res[fname].reset_index().to_excel('/Users/kadima/experiment_any/anomaly-detection/disagreement_matrix/engineering_version/auc_results/'+fname+"_auc_collection.xlsx",
                                header=['combination', 'auc'])

        print(fname, 'Done')
        print("="*50)

    print(main_detector_dict)










