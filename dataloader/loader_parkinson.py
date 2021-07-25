
import pandas as pd
import re

file_path = "c:/Users/froke/jupyter_notebook_files/anomaly_detection/EMEM算法/Parkinson_withoutdupl_75.txt"
def read_data(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()
    # print(data[28:-2])
    col_names = []
    list_ = []
    for id_, line in enumerate(data):
        if line.startswith("@ATTRIBUTE"):
            colName = re.findall("@ATTRIBUTE '(.*?)'", line)[0]
            col_names.append(colName)
        if id_ >= 28 and id_<= len(data) - 1 and line != "\n":
            new_line = line.split(",")
            new_line = [x.strip() for x in new_line]
            new_line = [float(new_line[x]) for x in range(len(new_line)-1)] + \
                       [new_line[-1].replace("'", "")]
            list_.append(new_line)

    df = pd.DataFrame(list_, columns=col_names)
    df['outlier'] = df.outlier.apply(lambda x: 1 if x =='yes' else 0)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y