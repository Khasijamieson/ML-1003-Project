#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re

def data_format(data):
    
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
