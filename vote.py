import os
import numpy as np
from sklearn.metrics import confusion_matrix
from stats import *
from collections import Counter


def load_data(path):
    res = {}
    with open(path, 'r') as file:
        lines = file.readlines()[1:]
    for line in lines:
        id_log_target = line.replace(',', '').replace('[', '').replace(']', '').split()
        idx = id_log_target[0]
        logits = list(map(float, id_log_target[1:4]))
        labels = id_log_target[4]
        preds = logits.index(max(logits))
        if idx in res:
            res[idx]['preds'].append(preds)
        else:
            res[idx] = {
                'logits': logits,
                'preds': [int(preds)],
                'labels': int(labels),
            }
    for k in res:
        counter = Counter(reversed(sorted(res[k]['preds'])))
        res[k]['vote'] = counter.most_common(1)[0][0]
    #     print(idx, logits, preds, labels)
    g_res = {}
    for k in res:
        key = k[:-2]
        if key not in g_res:
            g_res[key] = {
                'preds': [res[k]['vote']],
                'labels': res[k]['labels']
            }
        else:
            g_res[key]['preds'].append(res[k]['vote'])
    for k in g_res:
        counter = Counter(reversed(sorted(g_res[k]['preds'])))
        g_res[k]['vote'] = counter.most_common(1)[0][0]
    print(g_res)
    return g_res
        

def cal_CM(res):
    pred_label = np.array([[res[x]['vote'], res[x]['labels']] for x in res])
    preds = pred_label[:, 0]
    labels = pred_label[:, 1]
    cm = confusion_matrix(labels, preds)
    # print(cm)
    draw_cm(labels, preds)
    draw_score(labels, preds)
    drawdis(labels, preds)
    get_acc(labels, preds)
        

if __name__ == '__main__':
    res = load_data('/root/autodl-tmp/train_results/v2/vit_b_mixup_fold0_240305/val0.txt')
    cal_CM(res)