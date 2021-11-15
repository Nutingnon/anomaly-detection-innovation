import sys
import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self):
        data_path = "/Users/kadima/experiment_any/anomaly-detection/datasets_resend/"
        training_dict = dict()
        label_dict = dict()

        for root, path, files in os.walk(data_path):
            for file in files:
                if file.endswith("txt"):
                    if "label" not in file:
                        with open(root + file, 'r') as d:
                            data = d.readlines()
                            data = [x.split() for x in data]
                            data = [[float(i) for i in x] for x in data]
                        training_dict[file[:-9]] = np.asarray(data).astype(float)
                    else:
                        with open(root + file, 'r') as d:
                            label = d.readlines()
                            label = np.asarray([1 - int(x[0]) for x in label])
                        label_dict[file[:-10]] = label

        self.training_dict = training_dict
        self.label_dict = label_dict

    def get_train_data(self):
        return self.training_dict, self.label_dict



