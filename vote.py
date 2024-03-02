import os


def load_data(path):
    res = {}
    with open(path, 'r') as file:
        lines = file.readlines()[1:]
    for line in lines:
        id_log_target = line.replace(',', '').replace('[', '').replace(']', '').split()
        idx = id_log_target[0][:-2]
        logits = list(map(float, id_log_target[1:4]))
        labels = id_log_target[4]
        preds = logits.index(max(logits))
        if idx in res:
            res[idx]['preds'].append(preds)
        else:
            res[idx] = {
                'preds': [preds],
                'labels': labels
            }
        print(idx, logits, preds, labels)
    print(res)
        
        
        

if __name__ == '__main__':
    load_data('test_results/pd_finetune/0.txt')