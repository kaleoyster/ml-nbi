"""
Description:
    Preprocessing functions to run the models

Date:
    30th September, 2022
"""

import sys
import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import export_graphviz
from collections import defaultdict
from tqdm import tqdm
import pydotplus
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

def read_csv(csv_file):
    """
    Read csv files using pandas
    """
    _df = pd.read_csv(csv_file)
    return _df

def create_label(intervention_columns):
    """
    Return positive or negative function
    """
    labels = []
    for value in intervention_columns:
        if value == 0:
            label_val = 'negative'
            labels.append(label_val)
        else:
            label_val = 'positive'
            labels.append(label_val)
    return labels

def normalize(_df, columns):
    """
    Function for normalizing the data

    Args:
        _df (dataframe)
        columns (features)
    """
    for feature in columns:
        _df[feature] = _df[feature].astype(int)
        max_value = _df[feature].max()
        min_value = _df[feature].min()
        _df[feature] = (_df[feature] - min_value) / (max_value - min_value)
    return _df

def remove_null_values(_df):
    """
    Description: return a new df with null values removed
    Args:
        df (dataframe): the original dataframe to work with
    Returns:
        df (dataframe): dataframe
    """
    for feature in _df:
        if feature != 'structureNumber':
            try:
                _df = _df[~_df[feature].isin([np.nan])]
            except:
                print("Error: ", feature)
    return _df


def remove_duplicates(_df, column_name='structureNumbers'):
    """
    Description: return a new df with duplicates removed
    Args:
        df (dataframe): the original dataframe to work with
        column (string): columname to drop duplicates by
    Returns:
        newdf (dataframe)
    """
    temp = []
    for group in _df.groupby(['structureNumber']):
        structure_number, grouped_df = group
        grouped_df = grouped_df.drop_duplicates(subset=['structureNumber'],
                               keep='last'
                               )
        temp.append(grouped_df)
    new_df = pd.concat(temp)
    return new_df
