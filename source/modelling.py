# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:29:08 2021
@author: Debanjan

Applying Random Forest Regression model to predict hourly
SOC variation for the battery.
"""

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from source import preProcessing


def model_randomForest(data):
    """
    Random Forest regression model with hyper-parameter tuning for
    optimization.
    args: processed dataframe
    returns: optimized training model saved as a pickle file
    """

    df_processed = data

    X, y, X_train, X_test, y_train, y_test, X_test_copy = preProcessing.processing_step2(df_processed)

    # Hyper-parameter tuning

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(4, 8, num=5)]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 15]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    parameters = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    rfm = RandomForestRegressor()

    # Grid Search with K-fold Cross Validation
    grid_rf = RandomizedSearchCV(rfm, param_distributions=parameters, scoring='neg_root_mean_squared_error', cv=cv,
                                 return_train_score=True, n_iter=50, n_jobs=-1, random_state=20)

    # Fit the best model
    grid_rf.fit(X_train, y_train)

    # Fit the best model
    grid_rf.fit(X_train, y_train)

    # Obtain the accuracy and parameters for the best model
    print("\nModel: Random Forest Regressor")
    print("***********************************")
    print("Best parameters found: ", grid_rf.best_params_)
    print("Lowest RMSE found: ", np.abs(grid_rf.best_score_))

    # Clone and save the model to a pickle file
    model_rf = clone(grid_rf.best_estimator_)
    model_rf.fit(X_train, y_train)

    # Obtain the most important features
    print("\nTop features for the model\n")
    feat_importance = pd.Series(model_rf.feature_importances_, index=X.columns[0:len(X.columns)])

    print(feat_importance.nlargest(5))

    model_file = './model/rf_model.pkl'
    joblib.dump(model_rf, model_file)

    return model_file