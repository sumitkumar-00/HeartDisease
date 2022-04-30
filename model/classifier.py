import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import roc_auc_score
from joblib import dump
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

#Add util directory to path
curr_dir = sys.path[0]
parent_dir = Path(curr_dir).parents[0]
dir = os.path.join(parent_dir, 'util')
sys.path.append(dir)

from custom_transformer import NumericalFeatures, CategoricalFeatures

def load_file(file):
    """
    This function loads the data file into a Panda's data frame
    :param file: file name with path
    :return: DataFrame
    """
    df = pd.read_csv(file)
    print('{0} observations and {1} features successfully loaded'.format(df.shape[0], df.shape[1]))

    return df

def main():
    if len(sys.argv) < 2:
        print("Please provide path to the data file")
    else:
        file = sys.argv[1]

    try:
        df = load_file(sys.argv[1])
    except FileNotFoundError:
        print("File not found. Please provide correct filepath")

if __name__ == '__main__':
    main()
