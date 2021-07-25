import scipy.io
import os

def getData(fileName):
    # check fileName:
    files = [x for x in os.listdir("../EMEM算法/datasets/") if x.endswith(".mat")]
    mat = scipy.io.loadmat(f"../EMEM算法/datasets/{fileName}.mat")
    X = mat["X"]
    y = mat["y"]
    return X, y