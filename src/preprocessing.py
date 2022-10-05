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
from collections import defaultdict
from tqdm import tqdm
import pydotplus

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

def preprocess():
    csv_file = '../data/nebraska_deep.csv'
    df = read_csv(csv_file)

    # Remove null values:
    df = df.dropna(subset=['deck',
                           'substructure',
                           'deckNumberIntervention',
                           'subNumberIntervention',
                           'supNumberIntervention'
                          ])

    df = remove_duplicates(df)

    # Remove values encoded as N:
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]
    df = df[~df['scourCriticalBridges'].isin(['N', 'U', np.nan])]
    df = df[~df['deckStructureType'].isin(['N', 'U'])]

    # Fill the null values with -1:
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)

    df.toll.fillna(value=-1, inplace=True)
    df.designatedInspectionFrequency.fillna(value=-1, inplace=True)
    df.deckStructureType.fillna(value=-1, inplace=True)
    df.typeOfDesign.fillna(value=-1, inplace=True)

    # Normalize features:
    columns_normalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "avgDailyTruckTraffic",
                        "supNumberIntervention",
                        "subNumberIntervention",
                        "deckNumberIntervention",
                        "latitude",
                        "longitude",
                        "skew",
                        "numberOfSpansInMainUnit",
                        "lengthOfMaximumSpan",
                        "structureLength",
                        "bridgeRoadwayWithCurbToCurb",
                        "operatingRating",
                        "scourCriticalBridges",
                        "lanesOnStructure",
                        ]

    # Select final columns:
    columns_final = [
                    "deck",
                    "yearBuilt",
                    "superstructure",
                    "substructure",
                    "averageDailyTraffic",
                    "avgDailyTruckTraffic",
                    "material",
                    "designLoad",
                    "snowfall",
                    "freezethaw",
                    "supNumberIntervention",
                    "subNumberIntervention",
                    "deckNumberIntervention",

                    "latitude",
                    "longitude",
                    "skew",
                    "numberOfSpansInMainUnit",
                    "lengthOfMaximumSpan",
                    "structureLength",
                    "bridgeRoadwayWithCurbToCurb",
                    "operatingRating",
                    "scourCriticalBridges",
                    "lanesOnStructure",

                    "toll",
                    "designatedInspectionFrequency",
                    "deckStructureType",
                    "typeOfDesign",

                    "deckDeteriorationScore",
                    "subDeteriorationScore",
                    "supDeteriorationScore"
                ]

    cols = columns_normalize
    data_scaled = normalize(df, columns_normalize)
    X = data_scaled[columns_final]
    X = remove_null_values(X)

    deckLabels = X['deckDeteriorationScore']
    y = create_label(deckLabels)

    # Convert them into arrays
    X = np.array(X)
    y = np.array(y)

    return X, y, cols
