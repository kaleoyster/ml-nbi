"""
Description:
    Regression model to predict the future maintenance of bridges

Date:
   October 3rd, 2022
"""

import sys
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

def main():
    print("regression model")

main()
