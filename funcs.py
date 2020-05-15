#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from scipy import sparse


def data_format(data):


    '''
    Input: pandas dataframe of the raw data
    output: a list of dictionaries that represent each example. The key of each dictionary is an encoding for a unique word, and the value
           is the tfidf weight of that word in the example; another dictionary whose keys are the indices of an example, and the values a 
           list of labels associated with the respective example 
    '''    

    '''
    We found that some lines of the data have their first feature as their collection of labels
    We attribute this to an error in data collection/formatting on the part of whoever compiled this data
    To deal with this, we have to drop these rows. The labels that might go with them are unsalvageable to us
    We will therefor find all labels that contain a ':' character, as this is an indicator of the issue
    We will then use a pandas mask to remove these row indices
    '''
    
    ids_to_drop = [i for i in range(len(data)) if ':' in data.iloc[i]['labels'] ]
    #we must remember to reset the index as well.
    train = data.iloc[~data.index.isin(ids_to_drop)].reset_index(drop=True)
    
    
    '''
    This code block constructs a list of dictionaries. Each dictionary represents the 
    features column of one of the 15539 examples in the dataset
    '''
    feat_dicts = []
    for i in range(len(train)):
        line_dict = {}
        line = train['features'][i]
        keys = re.findall(r'(\d+):', line)
        values = re.findall(r'\d+:(\d+\.\d+)', line)
        for i in range(len(keys)):
            line_dict[int(keys[i])] = float(values[i])
        feat_dicts.append(line_dict)
        
    '''
    This code block constructs a dictionary
    Each key represents the index of an example in train
    The associatec value is a set (we have chosen set for ease of membership testing later on) which contains 
    all the labels that are associated with the corresponding example
    '''
    label_dict = {}
    for i in range(len(train)):
        labels = train['labels'][i]
        label_dict[i] = list(np.array(re.findall(r'(\d+)', labels)).astype('int'))
  

    return feat_dicts, label_dict

 
def sparsify(feat_dicts, label_dict):  

    '''
    Input: outputs of data_format; a list of dicts representing examples, and a dictionary representing the labels for those examples
    Output: sparse matrix representations of the two inputs
    '''

    x_s =sparse.lil_matrix((len(feat_dicts), 5000))
    for i in range(len(feat_dicts)):
        for j in list(feat_dicts[i].keys()):
            x_s[i,j] = feat_dicts[i][j]
   
  
    y_s=sparse.lil_matrix((len(label_dict),3993))
    for i in label_dict:
        for j in label_dict[i]:
            y_s[i,j] = 1

    return x_s, y_s


