import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import re
from sklearn.preprocessing import RobustScaler
import scipy.stats as ss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import scipy.io
from __future__ import division
from __future__ import print_function
import os
import sys
from time import time

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP


"""
Input
1. datasets: X
2. detectors: list
3. trial_nums: int
    the trial_nums control the iteratnion times of selecting a disagreement point
"""








