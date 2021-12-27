
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from tqdm import tqdm


def DrawROC(test_y,test_pred,logdir):
    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(test_y, test_pred, pos_label=1)
    # write ROC raw data
    with open(str(logdir) + "/best_test_roc.tsv", "w") as the_file:
        the_file.write("#thresholds\ttpr\tfpr\n")
        for t, tpr, fpr in zip(test_thresholds, test_tpr, test_fpr):
            the_file.write("{}\t{}\t{}\n".format(t, tpr, fpr))

    test_auc = metrics.auc(test_fpr, test_tpr)
    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(test_fpr, test_tpr, 'b',label='AUC = %0.2f' % test_auc)
    fig.savefig(str(logdir) + "/best_test_roc.png")
    plt.close(fig)
    return test_auc,test_fpr, test_tpr, test_thresholds


def DrawRecall_Pre_F1(test_y,test_pred,logdir):
    precision, recall, thresholds = metrics.precision_recall_curve(test_y, test_pred,pos_label=1)
    # write P-R raw data
    with open(str(logdir) + "/pre_recall.tsv", "w") as the_file:
        the_file.write("#thresholds\tprecision\trecall\n")
        for t, pre, rec in zip(thresholds, precision, recall):
            the_file.write("{}\t{}\t{}\n".format(t, pre, rec))

    fig = plt.figure()
    plt.title('Precisopn-Recall')
    plt.plot(recall, precision, 'b')
    fig.savefig(str(logdir) + "/pre_recall.png")
    plt.close(fig)

    fig = plt.figure()
    plt.title('thresholds-TPR')
    plt.plot(thresholds, recall[0:-1], 'b')
    fig.savefig(str(logdir) + "/thresholds_tpr.png")
    plt.close(fig)

    DrawF1score_CDF(precision, recall, logdir)


def DrawF1score_CDF(precision,recall,logdir):
    f1_scores = []
    f1_socre_percents = []
    CDF_X = list(np.linspace(0, 1, num=100))  # f1-score-cdf-x
    for i in range(len(precision)):
        f1_socre = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        f1_scores.append(f1_socre)
    for CDF in CDF_X:
        f1_socre_percents.append(GetPercent_Of_F1_score(f1_scores,CDF))
    fig = plt.figure()
    plt.title('F1score-CDF')
    plt.plot(CDF_X, f1_socre_percents, 'b')
    fig.savefig(str(logdir) + "/F1score-CDF.png")
    plt.close(fig)
    with open(logdir + "/F1score-CDF.tsv", "w") as the_file:
        the_file.write("#F1_score\tpercentage\n")
        for c, per in zip(CDF_X, f1_socre_percents):
            the_file.write("{}\t{}\n".format(c, per))


def GetPercent_Of_F1_score(f1_scores,CDF):
    num = 0
    for f1_score in f1_scores:
        if f1_score <= CDF:
            num += 1
    percent = float(num)/len(f1_scores)
    return percent


def Draw_ROC_K(similar_rate,truth,logdir):
    sort_similar,sort_truth = similar_truth_sort(similar_rate,truth)
    keylist = [i for i in range(5, len(truth), 5)]
    fpr_my,tpr_my = myself_roc(sort_similar,sort_truth,keylist)
    auc_my = metrics.auc(fpr_my,tpr_my)

    with open(str(logdir) + "/roc_k.tsv", "w") as the_file:
        the_file.write("#k\ttpr\tfpr\n")
        for k, tpr, fpr in zip(keylist, tpr_my, fpr_my):
            the_file.write("{}\t{}\t{}\n".format(k, tpr, fpr))
    fig = plt.figure()
    plt.title('roc_k')
    plt.plot(fpr_my, tpr_my, 'b')
    fig.savefig(str(logdir) + "/roc_k.png")
    plt.close(fig)


    with open(logdir + "/k_recall.tsv", "w") as the_file:
        the_file.write("#k\recall\n")
        for k, recall in zip(keylist, tpr_my):
            the_file.write("{}\t{}\n".format(k, recall))
    fig = plt.figure()
    plt.title('k_recall')
    plt.plot(keylist, tpr_my, 'b')
    fig.savefig(str(logdir) + "/k_recall.png")
    plt.close(fig)



def similar_truth_sort(similar,truth):
    sort_similar = []
    sort_truth = []
    sort_index = np.argsort(-similar) # from max to small
    for i in sort_index:
        sort_similar.append(similar[i])
        sort_truth.append(truth[i])
    return sort_similar,sort_truth

def myself_roc(similar,truth,keylist):
    fpr = []
    tpr = []
    for key in keylist:
        tp = float(0)
        fp = float(0)
        tn = float(0)
        fn = float(0)
        for i in range(key):
            if truth[i] == True:
                tp += 1
            else:
                fp += 1
        for i in range(key,len(similar)):
            if truth[i] == True:
                fn += 1
            else:
                tn += 1
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    return fpr,tpr
