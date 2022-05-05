import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import roc_auc_score
from joblib import dump
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

# Add util directory to path
curr_dir = sys.path[0]
parent_dir = Path(curr_dir).parents[0]
dir = os.path.join(parent_dir, 'util')
sys.path.append(dir)

from custom_transformer import NumericalFeatures, CategoricalFeatures


def load_file(file):
    """
    This function loads the data file into a Panda's data frame
    :param file: file name with path
    :return: Features and the dependent variable i.e HeartDisease column
    """
    df = pd.read_csv(file)
    print('{0} observations and {1} features successfully loaded'.format(df.shape[0], df.shape[1]))

    y = df['HeartDisease']
    y = [1 if x == 'Yes' else 0 for x in y]
    X = df.drop(['HeartDisease'], axis=1)
    return X, y


def build_pipeline():
    pipeline_numerical = Pipeline(steps=
                                  [('Numerical_Features', NumericalFeatures()),
                                   ('StandardScaler', StandardScaler())
                                   ])

    pipeline_categorical = Pipeline(steps=
                                    [('Categorical_Features', CategoricalFeatures()),
                                     ('OneHotEncoder', OneHotEncoder())
                                     ])

    pre_processing_pipeline = FeatureUnion(
                                 [('Numerical', pipeline_numerical),
                                  ('Categorical', pipeline_categorical)
                                  ])

    return pre_processing_pipeline


def check_model_accuracy(name, model, pre_processing_pipeline, param_grid, X_train, X_test, y_train, y_test):
    print("Fitting " + name)
    pipeline = Pipeline(steps=
                        [('SMOTENC', SMOTENC(categorical_features=list(X_train.dtypes=='object'))),
                         ('preprocessing_pipeline', pre_processing_pipeline),
                         (model[0], model[1])
                         ])
    gs = GridSearchCV(pipeline, cv=3, param_grid=param_grid, scoring='roc_auc')
    gs.fit(X_train, y_train)
    y_probs = gs.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_probs)
    print(name + " auc_score is " + str(auc_score))
    result = [name, gs.best_estimator_, auc_score]
    return result

def main():
    if len(sys.argv) < 2:
        print("Please provide path to the data file")
    else:
        file = sys.argv[1]

    try:
        X, y = load_file(sys.argv[1])
    except FileNotFoundError:
        print("File not found. Please provide correct filepath")

    # Split train and test data
    random_state = 7
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    # Random Forest Classifier
    model = ('rf', RandomForestClassifier())
    param_grid = {
        'rf__criterion': ['gini'],
        'rf__max_depth': [2],
        'rf__random_state': [random_state],
        'rf__n_estimators': [100]
    }
    result = check_model_accuracy('RandomForestClassifier', model, build_pipeline(), param_grid, X_train, X_test, y_train, y_test)
    results.append(result)

    # GradientBoosting Classifier
    model = ('gboost', GradientBoostingClassifier())
    param_grid = {
        'gboost__learning_rate': [0.01, 0.1],
        'gboost__max_depth': [3, 5],
        'gboost__random_state': [random_state]
    }
    result = check_model_accuracy('GradientBoostingClassifier', model, build_pipeline(), param_grid, X_train, X_test, y_train, y_test)
    results.append(result)

    # XGBClassifier
    model = ('xgb', XGBClassifier())
    param_grid = {
        'xgb__learning_rate': [0.05, 0.1]
    }
    result = check_model_accuracy('XGBClassifier', model, build_pipeline(), param_grid, X_train, X_test, y_train, y_test)
    results.append(result)

    # KNNClassifier
    model = ('KNN', KNeighborsClassifier())
    param_grid = {
        "KNN__n_neighbors": [5, 10]
    }
    result = check_model_accuracy('KNN', model, build_pipeline(), param_grid, X_train, X_test, y_train, y_test)
    results.append(result)

    # Conclusion
    conclusion(results)


def conclusion(results):
    auc_scores = [result[2] for result in results]
    best_model_index = auc_scores.index(max(auc_scores))
    print(results[best_model_index][0] + " is the best performing model with auc_score of " + str(max(auc_scores)))
    dump(results[best_model_index][1], 'model/loan_data.pkl')
    print("Saving best model as a pickled file")


if __name__ == '__main__':
    main()
