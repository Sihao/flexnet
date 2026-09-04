#!/Users/donyin/miniconda3/envs/imperial/bin/python

import pandas
from pathlib import Path
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import (
    KNeighborsRegressor,
)  # LinearRegression, RandomForestRegressor, or KNeighborsRegressor.


def make_imputations_for_single_col(dataframe, colname, num_imputations=5):
    """
    take a dataframe and a column name, and:
    - remove the missing rows in the target column
    - remove columns via various rules
    - one-hot encode the non-numeric columns
    - impute the missing values
    - return n imputed dataframes
    """
    # [0] keep a record of the dropped columns and the reason for dropping
    column_bank = {}

    # [1] drop all the raws where the target column is missing
    data = dataframe.copy()
    data = data.dropna(subset=[colname])

    # [2] drop all columns where each value is unique, e.g., each value is a subject id
    dropped = []
    for column in data.columns:
        if len(data[column].unique()) == len(
            data
        ) and not pandas.api.types.is_numeric_dtype(data[column]):
            data = data.drop(column, axis=1)
            dropped.append(column)
    column_bank.update({"dropped because of each value is unique categorical": dropped})

    # [3] drop the columns where there is only 1 value
    dropped = []
    for column in data.columns:
        if len(data[column].unique()) == 1:
            data = data.drop(column, axis=1)
            dropped.append(column)
    column_bank.update({"dropped because of it has only 1 value everywhere": dropped})

    # [4] drop the columns where the missing data is more than 50%
    dropped = []
    for column in data.columns:
        missing_data_ratio = data[column].isnull().sum() / len(data)
        if missing_data_ratio > 0.5:
            data = data.drop(column, axis=1)
            dropped.append(column)
    column_bank.update({"dropped because of missing data ratio > 50%": dropped})

    # [5] recode non-numeric columns one-hot
    non_numeric_columns = []
    for column in data.columns:
        if not pandas.api.types.is_numeric_dtype(data[column]):
            non_numeric_columns.append(column)
    data = pandas.get_dummies(data, columns=non_numeric_columns)

    # [final] report the droppings and the reasons
    print(column_bank)

    # [ ---- begin imputation ---- ]
    # [1] calculate the number of imputations we need by Rubin's rule
    """Instead of filling in a single value for each missing value, Rubin's (1987) multiple imputation procedure replaces each missing value with a set of plausible values that represent the uncertainty about the right value to impute."""
    missing_data_ratio = data.isnull().sum().sum() / np.prod(data.shape)
    missing_data_percentage = round(missing_data_ratio * 100)
    num_imputations = max(5, missing_data_percentage)
    print(
        f"Missing data percentage: {missing_data_percentage}% / Number of Imputations: {num_imputations}"
    )

    # [2] fix the data type
    """During the imputation, the imputation needs to know which columns are float and which are int as we don't want to assign float values with several decimal points to int columns. So we need to define the float and int columns here. float columns is defined as when dropping the NA values, the unique values have at least 1 have a decimal point. Note that this imputation method here is currently assuming all the values to be float and round them to int at the end. It is unclear to what extent this method is appropriate in this case."""
    float_columns = [
        col
        for col in data.columns
        if data[col].dropna().apply(lambda x: int(x) != x).any()
    ]
    int_columns = data.columns.difference(float_columns)
    data = data.astype(
        float
    )  # make all float and turn the int columns back to int at the end

    # [3] impute the data
    """Note that initially the n_neighbors is set to 5 but it does not converge at all. The current value of 16 is experimented with and it seems to work. But it is unclear how to choose this value systematically"""
    inputed_dataframes = []
    for i in range(num_imputations):
        imputer = IterativeImputer(
            estimator=KNeighborsRegressor(
                n_neighbors=16
            ),  # Adjust n_neighbors as needed
            missing_values=np.nan,
            sample_posterior=False,  # Not used with KNeighborsRegressor
            max_iter=1000,
            tol=1e-5,
            n_nearest_features=None,
            initial_strategy="mean",
            imputation_order="ascending",  # from features with fewest missing values to most.
            skip_complete=False,
            min_value=-np.inf,
            max_value=np.inf,
            verbose=1,
            random_state=i,
            add_indicator=False,
        )

        imputed_data = imputer.fit_transform(data)
        imputed_data_df = pandas.DataFrame(imputed_data, columns=data.columns)

        # round the int columns
        for column in int_columns:
            imputed_data_df[column] = imputed_data_df[column].round(0).astype(int)

        inputed_dataframes.append(imputed_data_df)

    return inputed_dataframes


if __name__ == "__main__":
    path_data = Path("results/3_experiment_212/results.csv")
    data = pandas.read_csv(path_data)
    imputed_dataframes = make_imputations_for_single_col(
        data, "num_spatial_attention_block"
    )
    print(imputed_dataframes)
