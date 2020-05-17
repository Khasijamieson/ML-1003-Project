#!/usr/bin/env python
# coding: utf-8

# ### Parsing and Baseline

# **Importing Libraries**

# In[328]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import sparse
import funcs
from skmultilearn.adapt import BRkNNaClassifier
from skmultilearn.adapt import MLkNN
import matplotlib.pyplot as plt

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
import time

from sklearn.metrics import label_ranking_average_precision_score


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[315]:


data = pd.read_csv('../train.csv')
val = pd.read_csv('../dev.csv')


# In[316]:


#Train formatting
x_train, y_train = funcs.data_format(data)
#Turns dicts into sparse matrices
x_train_s, y_train_s = funcs.sparsify(x_train, y_train)


# In[318]:


#same for val
x_val, y_val = funcs.data_format(val)
x_val_s, y_val_s = funcs.sparsify(x_val, y_val)


# In[327]:


start=time.time()
classifier = BinaryRelevance(
    classifier = RandomForestClassifier(),
    require_dense = [False, True]
)

classifier.fit(x_train_s, y_train_s)

print('training time taken: ',round(time.time()-start,0),'seconds')


# In[ ]:


parameters = {'max_depth': [10, 50, 100, 150], 
              'min_samples_split': [4, 8, 16, 32],
              'min_samples_leaf': [1, 2, 3],
             }

start=time.time()

classifier = GridSearchCV(RandomForestClassifier(), parameters, scoring=label_ranking_average_precision_score)
classifier.fit(x_train, y_train)

print('training time taken: ',round(time.time()-start,0),'seconds')
print('best parameters :', classifier.best_params_, 'best score: ',
      clf.best_score_)

