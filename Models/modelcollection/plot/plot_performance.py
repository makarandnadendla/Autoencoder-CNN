import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns

def PlotCm(y_true, y_pred, target_names, output_path, figsize=(13,10)):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=np.unique(y_true.argmax(axis=1)))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax, cbar=False)
    plt.savefig(output_path)

def PlotCr(y_true, y_pred, target_names, output_path, figsize=(11,5)):
    class_report = classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names, output_dict = True)
    results_pd = pd.DataFrame(class_report).iloc[:-1,:].transpose()
    results_pd = results_pd.append(pd.Series([None,None,None], index = ["precision", "recall", "f1-score"] ,name = ""))
    results_pd = results_pd.reindex(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "", "accuracy", "macro avg", "weighted avg"])
    plt.figure(figsize = figsize)
    plt.rcParams.update({'font.size': 13})
    colormap = ListedColormap(["white"])
    sns.heatmap(results_pd, annot = True, cmap = colormap, cbar = False)
    plt.savefig(output_path)