import random
import numpy as np
import scipy.stats as ss
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def drawdis(labels, preds):
    data = [labels, preds]
    A = ['GT', 'Pred']
    fig, axes = plt.subplots(1, 2,figsize=(12, 6))
    for i in range(2):
        v, c = np.unique(data[i], return_counts=True)
        v = [f'UPDRS-{x}' for x in v]
        axes[i].bar(v, c, width=0.5, align='center')
        for a, b in zip(v, c):
            axes[i].text(a, b, b, ha='center', va='bottom')
        axes[i].axis(ymin=0, ymax=100)
        axes[i].set_ylabel('Counts')
        axes[i].set_xlabel('UPDRS')
        axes[i].set_title(A[i])
    plt.savefig('img_st/his.png')
    

def cal_xyc(i, j):
    random_range = [(-0.125, 0.125), (0.875, 1.125), (1.875, 2.125)]
    cmap = ['g', 'y', 'r']
    x = random.uniform(*random_range[i])
    y = random.uniform(*random_range[j])
    c = cmap[int(abs(i - j))]
    return x, y, c

def draw_score(labels, preds):
    ax = plt.figure(figsize=(6, 6)).gca()
    
    ## fig color
    ax.set_facecolor('#e0e0e0')
    for i in range(labels.shape[0]):
        a, b, c = cal_xyc(preds[i], labels[i])
        plt.scatter(a, b, color=c)
        
    ## set integer
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ## remove border
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ## remove tick lines
    ax.tick_params(axis='both', which='both', length=0)

    ## withe grid
    plt.grid(True, color='white')
    plt.xlabel('Ratings from Model')
    plt.ylabel('Ratings from Experts')
    plt.savefig('img_st/sc.png')


def draw_cm(labels, preds):
    actual_labels = labels
    predicted_labels = preds
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    # Normalize the confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix_normalized, cmap=plt.cm.Blues)

    # Add the text annotations for counts and normalized values.
    for (i, j), val in np.ndenumerate(conf_matrix):
        norm_val = conf_matrix_normalized[i, j]
        ax.text(j, i, f'{val}\n({norm_val:.2f})', ha='center', va='center', color='white' if norm_val > 0.5 else 'black')

    # Add colorbar
    fig.colorbar(cax)

    # Set labels for axes
    ax.set_xlabel('Ratings from Model')
    ax.set_ylabel('Ratings from Experts')

    # Set tick marks for grid
    ax.set_xticks(np.arange(conf_matrix.shape[1]))
    ax.set_yticks(np.arange(conf_matrix.shape[0]))

    # Set tick labels for grid
    ax.set_xticklabels(np.arange(conf_matrix.shape[1]))
    ax.set_yticklabels(np.arange(conf_matrix.shape[0]))
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig('img_st/cm.png')


def get_acc(labels, preds):
    classification_rep = classification_report(labels, preds)
    print('classification_rep:')
    print(classification_rep)


def cal_spearman(x, y):
    r = ss.spearmanr(x, y)[0]
    print(f"spearman'r:{r}")


if __name__ == '__main__':
    x = np.load('train_results/dwpose128_stgan_pe_proj/0/labels.npy')
    y = np.load('train_results/dwpose128_stgan_pe_proj/0/preds.npy')

    drawdis(x, y)
    draw_score(x, y)
    draw_cm(x, y)
    cal_spearman(x, y)
    get_acc(x, y)