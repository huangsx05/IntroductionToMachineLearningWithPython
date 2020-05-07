#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install mglearn


# In[30]:


import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[31]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
#print(y)
# we rename the classes "blue" and "red" for illustration purposes
y_named = np.array(["blue", "red"])[y] # 0-blue, 1-red
#print(y_named)
# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner
X_train, X_test, y_train_named, y_test_named, y_train, y_test =     train_test_split(X, y_named, y, random_state=0)
# build the gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)


# In[32]:


print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(
    gbrt.decision_function(X_test).shape))


# In[33]:


# show the first few entries of decision_function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))


# In[34]:


# recover the prediction by looking only at the sign of the decision function
print("Thresholded decision function:\n{}".format(
    gbrt.decision_function(X_test) > 0))
print("Predictions:\n{}".format(gbrt.predict(X_test)))


# In[35]:


## For binary classification, the “negative” class is always the first entry of the classes_
##attribute, and the “positive” class is the second entry of classes_. So if you want to
##fully recover the output of predict, you need to make use of the classes_ attribute:
    
# make the boolean True/False into 0 and 1
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int) # True=1, False=0
print("greater_zero: {}".format(greater_zero))
# use 0 and 1 as indices into classes_
pred = gbrt.classes_[greater_zero]
print("Pred:\n{}".format(pred))
# pred is the same as the output of gbrt.predict
print("pred is equal to predictions: {}".format(
np.all(pred == gbrt.predict(X_test))))


# In[36]:


## The range of decision_function can be arbitrary, and depends on the data and the
## model parameters:
decision_function = gbrt.decision_function(X_test)
print("Decision function minimum: {:.2f} maximum: {:.2f}".format(
    np.min(decision_function), np.max(decision_function)))


# In[40]:


## plot the decision_function
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 第一个画板 axes[0]，画decision boundary
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                fill=True, cm=mglearn.cm2)
# # 第一个画板 axes[1]，画decision function
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                            alpha=.4, cm=mglearn.ReBl)

for ax in axes:
    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                            markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                            markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    
cbar = plt.colorbar(scores_image, ax=axes.tolist())  
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
                "Train class 1"], ncol=4, loc=(.1, 1.1))


# In[42]:


## Predicting Probabilities (n各类别probability和为1)
print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(
    gbrt.predict_proba(X_test[:6])))


# In[43]:


## show the decision boundary on the
## dataset, next to the class probabilities for the class 1
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

mglearn.tools.plot_2d_separator(
    gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(
    gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                            markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                            markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
                "Train class 1"], ncol=4, loc=(.1, 1.1))


# In[ ]:




