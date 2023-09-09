from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    # print("Input train data shape: ", train_df.shape)
    # print("Input val data shape: ", val_df.shape)
    # print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #
    #       NAME_CONTRACT_TYPE ['Cash loans' 'Revolving loans']
    #       FLAG_OWN_CAR ['Y' 'N']
    #       FLAG_OWN_REALTY ['N' 'Y']
    #       EMERGENCYSTATE_MODE [nan 'No' 'Yes']
    #
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing. OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    # Encode string categorical features
    ordinal_cols = []
    one_hot_cols = []

    for col in working_train_df.select_dtypes(include="object").columns:
        if working_train_df[col].nunique() == 2:
            ordinal_cols.append(col)
        else:
            one_hot_cols.append(col)

    # Initialize ordinal encoder and fit to working_train_df
    ordinal_enc = OrdinalEncoder()
    ordinal_enc.fit(working_train_df[ordinal_cols])

    # Apply ordinal encoding to all dataframes
    working_train_df[ordinal_cols] = ordinal_enc.transform(
        working_train_df[ordinal_cols]
    )
    working_val_df[ordinal_cols] = ordinal_enc.transform(working_val_df[ordinal_cols])
    working_test_df[ordinal_cols] = ordinal_enc.transform(working_test_df[ordinal_cols])

    # Initialize one-hot encoder and fit to working_train_df
    one_hot_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_enc.fit(working_train_df[one_hot_cols])

    # Apply one-hot encoding to all dataframes
    train_one_hot = one_hot_enc.transform(working_train_df[one_hot_cols])
    val_one_hot = one_hot_enc.transform(working_val_df[one_hot_cols])
    test_one_hot = one_hot_enc.transform(working_test_df[one_hot_cols])

    # Replace original columns with one-hot encoded columns
    working_train_df = working_train_df.drop(columns=one_hot_cols).join(
        pd.DataFrame(train_one_hot)
    )
    working_val_df = working_val_df.drop(columns=one_hot_cols).join(
        pd.DataFrame(val_one_hot)
    )
    working_test_df = working_test_df.drop(columns=one_hot_cols).join(
        pd.DataFrame(test_one_hot)
    )

    print(working_train_df.shape, working_val_df.shape, working_test_df.shape)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    # Median Imputation
    median_imputer = SimpleImputer(strategy="median", missing_values=np.nan)
    median_imputer.fit(working_train_df.values)

    # Transform
    working_train_df = median_imputer.transform(working_train_df.values)
    working_val_df = median_imputer.transform(working_val_df.values)
    working_test_df = median_imputer.transform(working_test_df.values)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    scaler = MinMaxScaler()
    scaler.fit(working_train_df)
    working_train_df = scaler.transform(working_train_df)
    working_val_df = scaler.transform(working_val_df)
    working_test_df = scaler.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df
