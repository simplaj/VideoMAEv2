import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


def findAllFile(base, ext=None):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            if not ext is None and not fullname.endswith(ext):
                continue
            file_path.append(fullname)
    return file_path


def gen(paths):
    all_csv = {
        'train': [],
        'val': [],
        'test': []
    }
    labels = ['hy', 'slight', 'mild']
    for path in paths:
        mode, label, name = path.split('/')[-3:]
        all_csv[mode].append(f'{path} {labels.index(label)}\n')
    # print(all_csv)
    for key in all_csv:
        with open(f'{key}.csv', 'w') as file:
            random.shuffle(all_csv[key])
            file.writelines(all_csv[key])


def cal_label(paths):
    all_csv = []
    labels = ['hy', 'slight', 'mild']
    res = []
    for path in paths:
        mode, label, name = path.split('/')[-3:]
        all_csv.append(f'{path} {labels.index(label)}\n')
        res.append([path, labels.index(label)])
    return np.array(res)

            
def rechoose(data):
    data = sorted(list(set([x[:-6] for x in data])))
    res = cal_label(data)
    X, y = res[:, 0], res[:, 1]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    [print(np.sum(x=='0'),np.sum(x=='1'),np.sum(x=='2')) for x in [y_train, y_val, y_test]]
    with open('train.csv', 'w') as file:
        lines = []
        for data, label in zip(X_train, y_train):
            for i in range(6):
                lines.append(f'{data}_{i}.mp4 {label}\n')
        random.shuffle(lines)
        file.writelines(lines)
    with open('val.csv', 'w') as file:
        lines = []
        for data, label in zip(X_val, y_val):
            for i in range(6):
                file.write(f'{data}_{i}.mp4 {label}\n')
    with open('test.csv', 'w') as file:
        lines = []
        for data, label in zip(X_test, y_test):
            for i in range(6):
                file.write(f'{data}_{i}.mp4 {label}\n')
        

if __name__ == '__main__':        
    root = '/root/autodl-tmp/videos_all/'
    paths = findAllFile(root)
    rechoose(paths)