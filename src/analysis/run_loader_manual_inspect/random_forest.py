#!/Users/donyin/miniconda3/envs/imperial/bin/python

import numpy as np

from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def random_forest_select_features(
    dataframes: list[DataFrame], colname: str, fold: int = 5
):
    # [0] init banks
    importances_track = []
    r2_scores = []

    for dataframe in dataframes:
        try:
            X: DataFrame = dataframe.drop([colname], axis=1)
            y: DataFrame = dataframe[colname]
        except KeyError:
            first_associated_col = [
                col for col in dataframe.columns if col.startswith(colname)
            ][0]
            X: DataFrame = dataframe.drop(
                [col for col in dataframe.columns if col.startswith(colname)], axis=1
            )
            y: DataFrame = dataframe[first_associated_col]
            print(
                f"Column {colname} not found in the dataframe. Treating it as one-hot {first_associated_col} instead."
            )

        # [1] split the data randomly into X_train, X_test, y_train, y_test
        # Here, the test_size is derived from the fold variable. For example, a fold of 5 implies a 20% test set.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 / fold), random_state=0
        )

        # [2] fit the model
        if y_train.dtype == "object":
            clf = RandomForestClassifier(n_estimators=1000, random_state=0)
        if y_train.dtype in ["float", "float64", "int", "int64"]:
            clf = RandomForestRegressor(n_estimators=1000, random_state=0)

        clf.fit(X_train, y_train)

        # [3] predict
        y_pred = clf.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))

        # [4] get the importances
        importances = clf.feature_importances_
        importances_track.append(importances)

    # [5] get the mean importances
    mean_importances = np.mean(importances_track, axis=0)
    mean_r2_scores = np.mean(r2_scores)

    # [6] creating a dictionary with feature names and their mean importance scores
    feature_names = (
        X.columns
    )  # Assuming all DataFrames have the same columns minus the target
    importances_dict = {
        name: score for name, score in zip(feature_names, mean_importances)
    }
    importances_dict = dict(
        sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)
    )

    return importances_dict, mean_r2_scores
