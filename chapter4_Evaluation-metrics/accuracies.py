#!/usr/bin/env python
# coding: utf-8

# ## Accuracy implementation (simplest)

# In[1]:


l1 = [0,1,1,1,0,0,0,1]    #true targets
l2 = [0,1,0,1,0,1,0,0]    #predicted targets


# In[2]:


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


# In[3]:


print(accuracy(l1, l2))


# ## accuracy using scikit-learn

# In[4]:


from sklearn import metrics
print(metrics.accuracy_score(l1, l2))


# ## accuracy calculation with True positive, True negative, False positive, False negative

# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


print(true_positive(l1, l2))


# In[10]:


print(false_positive(l1, l2))


# In[11]:


print(false_negative(l1, l2))


# In[12]:


print(true_negative(l1, l2))


# ### Accuracy Score = (TP + TN) / (TP + TN + FP + FN)

# In[13]:


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


# In[14]:


print(accuracy_v2(l1, l2))


# In[15]:


print(accuracy(l1, l2))


# In[16]:


print(metrics.accuracy_score(l1, l2))


# ## Bingo!!!

# In[ ]:




