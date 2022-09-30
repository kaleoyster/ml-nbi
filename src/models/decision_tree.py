"""
Description:
    Decision tree model to predict future maintenance of bridges

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

def read_csv(csv_file):
    """
    Read csv files using pandas
    """
    _df = pd.read_csv(csv_file)
    return _df

def normalize(_df, columns):
    """
    Function for normalizing the data
    """
    print("Accessed the feature!")
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

def main():
    csv_file = '../../data/nebraska_deep.csv'
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

    data_scaled = normalize(df, columns_normalize)
    data_scaled = data_scaled[columns_final]
    data_scaled = remove_null_values(data_scaled)

    # Modeling
    kmeans_kwargs = {
                        "init": "random",
                        "n_init": 10,
                        "max_iter": 300,
                        "random_state": 42,
                    }

    list_of_parameters = ['supNumberIntervention',
                          'subNumberIntervention',
                          'deckNumberIntervention'
                      ]

    #dataScaled, lowestCount, centroids, counts = kmeans_clustering(data_scaled,
    #                                                               list_of_parameters,
    #                                                               kmeans_kwargs,
    #                                                               state=state)

main()
