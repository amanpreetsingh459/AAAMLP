#!/usr/bin/env python
# coding: utf-8

# ## Accuracy implementation (simplest)

l1 = [0,1,1,1,0,0,0,1]    #true targets
l2 = [0,1,0,1,0,1,0,0]    #predicted targets

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    # initialize a simple counter for correct predictions
    correct_counter = 0
    # loop over all elements of y_true
    # and y_pred "together"
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            # if prediction is equal to truth, increase the counter
            correct_counter += 1
    # return accuracy
    # which is correct predictions over the number of samples
    return correct_counter / len(y_true)

print("Accuracy V1:", accuracy(l1, l2))      #0.625


# ## accuracy using scikit-learn

from sklearn import metrics
print("Accuracy using sklearn:", metrics.accuracy_score(l1, l2))      #0.625


# ## accuracy calculation with True positive, True negative, False positive, False negative

def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true positives
    """
    # initialize
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true negatives
    """
    # initialize
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    """
    Function to calculate False Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positives
    """
    # initialize
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    """
    Function to calculate False Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false negatives
    """
    # initialize
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


print("Number of true positive:", true_positive(l1, l2))      #0.625

print("Number of false positive:", false_positive(l1, l2))      #0.625

print("Number of false negative:", false_negative(l1, l2))      #0.625

print("Number of true negative:", true_negative(l1, l2))      #0.625

# ### Accuracy Score = (TP + TN) / (TP + TN + FP + FN)

def accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy using tp/tn/fp/fn
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score


print("Accuracy V2:", accuracy_v2(l1, l2))      #0.625

print("Accuracy V1:", accuracy(l1, l2))      #0.625

print("Accuracy using sklearn:", metrics.accuracy_score(l1, l2))      #0.625

# ###Precision = TP / (TP + FP)

def precision(y_true, y_pred):
    """
    Function to calculate precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: precision score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision
    # ## Bingo!!!

print("Precision:", precision(l1, l2))      #0.6666666666666666

# ###Recall = TP / (TP + FN)

def recall(y_true, y_pred):
    """
    Function to calculate recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

print("Recall:", recall(l1, l2))      #0.5


# ###F1 = 2PR / (P + R)
# ###F1 = 2TP / (2TP + FP + FN)

def f1(y_true, y_pred):
    """
    Function to calculate f1 score
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: f1 score
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    score = 2 * p * r / (p + r)
    return score

print("F1 score:", f1(l1, l2))      #0.5714285714285715

from sklearn import metrics
print("F1 score through sklearn:", metrics.f1_score(l1, l2))      #0.5714285714285715


# ###TPR(True Positive Rate) = TP / (TP + FN)
# ###True Positive Rate==recall==sensitivity

def tpr(y_true, y_pred):
    """
    Function to calculate tpr
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: tpr/recall
    """
    return recall(y_true, y_pred)


# ###FPR(False Positive Rate) = FP / (TN + FP)
# ###(1 - FPR) = specificity == True Negative Rate(TNR)

def fpr(y_true, y_pred):
    """
    Function to calculate fpr
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: fpr
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp / (tn + fp)


# ### (Area Under ROC Curve) = (Area Under Curve) = AUC

l3 = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
l4 = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
		0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
		0.85, 0.15, 0.99]

from sklearn import metrics
print("AUC:", metrics.roc_auc_score(l3, l4))


# ###selecting best threshold value using ROC curve

# empty lists to store true positive and false positive values
tpr_list = []
fpr_list = []

# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

# some handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

# loop over all thresholds
for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    # calculate tpr
    temp_tpr = tpr(y_true, temp_pred)
    # calculate fpr
    temp_fpr = fpr(y_true, temp_pred)
    # append tpr and fpr to lists
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)

print("Threshold list:", thresholds)
print("TPR list:", tpr_list)
print("FPR list:", fpr_list)

import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()


# ### multi-class precision, and f1-score calculation using sklearn
from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

print("Multi-class precision with macro average:", metrics.precision_score(y_true, y_pred, average="macro"))
print("Multi-class precision with micro average:", metrics.precision_score(y_true, y_pred, average="micro"))
print("Multi-class precision with weighted average:", metrics.precision_score(y_true, y_pred, average="weighted"))

print("Multi-class f1-score with weighted average:", metrics.f1_score(y_true, y_pred, average="weighted"))

# ###confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
# some targets
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
#some predictions
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
# get confusion matrix from sklearn
cm = metrics.confusion_matrix(y_true, y_pred)
# plot using matplotlib and seaborn
plt.figure(figsize=(10, 10))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,
as_cmap=True)
sns.set(font_scale=2.5)
sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
plt.ylabel('Actual Labels', fontsize=20)
plt.xlabel('Predicted Labels', fontsize=20)

