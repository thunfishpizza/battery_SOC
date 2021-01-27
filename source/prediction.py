# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:29:08 2021
@author: Debanjan

Predict the SOC based on the fitted model.
"""
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from source import preProcessing


def predict_func(testFile, model='./model/rf_model.pkl'):
    """
    Prediction function on a test set
    args1: test data
    args2: the trained model (pickled)
    returns: None
    """

    print("\nRunning Prediction on test data")
    print("************************************")

    df_processed = testFile
    X, y, X_train, X_test, y_train, y_test, X_test_copy = preProcessing.processing_step2(df_processed)

    # Load the trained model
    loaded_model = joblib.load(model)

    # Apply prediction on test data
    y_pred = loaded_model.predict(X_test)

    # Apply prediction on train data
    y_pred_train = loaded_model.predict(X_train)

    print("\nTrain set error (RMSE): ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print("\nTest set error (RMSE): ", np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\nTrain set explained variance (R^2): ", r2_score(y_train, y_pred_train))
    print("\nTest set explained variance (R^2): ", r2_score(y_test, y_pred))

    # Create a new dataframe with the target and predictions and their corresponding instances
    submission = pd.DataFrame({
        "Instant": X_test_copy.index,
        "Actual": y_test,
        "Predicted": y_pred
    })

    # Save the submission dataframe to a csv file
    submission.to_csv('./data/output/predictions'+str(datetime.datetime.now())+'.csv', index=False)

    print("\nOutput saved as submission.csv")