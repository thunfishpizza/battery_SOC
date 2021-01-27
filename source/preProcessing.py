# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:29:08 2021
@author: Debanjan

Feature Engineering on raw data
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def processing_step1(data):
    """
    1. Remove outliers
    2. Add avg. temperature, charge and capacity
    3. Aggregate to hourly samples
    args: raw dataframe
    returns: transformed dataframe
    """

    df = data

    print("Processing Raw Dataframe\n")
    print("Column names and datatypes:", df.dtypes, sep='\n')

    print("\nRemoving outliers....\n")

    Q1 = df.quantile(0.25)  # 25th percentile
    Q3 = df.quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Inter-quartile range.

    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    print("\nHourly aggregation....\n")
    df = df.resample('60min').mean()
    df = df.dropna(how='all', axis=0)
    print(df.head())

    print("\nCreating avg. temp, energy and charge as new features....\n")
    df['T_avg'] = (df['T_max'] + df['T_min']) / 2  # Average temperature
    df['E'] = df['V'] + df['I']  # Energy (Wh)
    df['Q'] = df['E'] / df['V']  # Capacity (Ah)

    print("\nDropping redundant columns....\n")
    df = df.drop(['T_max', 'T_min'], axis=1)

    print("Final Feature list + target columns\n")
    print(df.head())
    print("\nSaving processed data to csv....\n")
    df.to_csv('./data/processed/data_processed.csv', index=False)
    print("---------------------------------------------------------------\n")

    df_processed = df.copy()
    return df_processed


def processing_step2(data):
    """
    1. Split the data into train and test set
    2. Normalize the train and test data
    args: feature engineered dataframe
    returns: training and testing data
    """
    df_processed = data

    # Divide the data into features and target
    X = df_processed.drop(['SOC'], axis=1)
    y = df_processed['SOC']

    print("\nNumber of features...\n")
    print(X.shape)

    # Split the data into 80% train and 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # Make a copy of X_test for later use
    X_test_copy = X_test.copy()

    # Normalize train and test set
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_test_copy
