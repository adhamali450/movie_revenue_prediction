from operator import mod

from sklearn.impute import SimpleImputer
from test_preprocessing import *
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn import metrics
import joblib


def main(model_no):

    filename = ''
    data = None

    #fetching train columns
    with open('../dumm.txt', 'r') as f:
        fetched_dumm = f.read().splitlines()

    with open('../ord_cols.txt', 'r') as f:
        fetched_ord_cols = f.read().splitlines()
    
    # regression
    if model_no in [1, 2]:
        data = merged = merge_samples_directors('./Milestone 1/movies-test-samples.csv',
                                                './Milestone 1/movie-director.csv')
        if model_no == 1:
            filename = './model1_ridge.sav'
        else:
            filename = './model2_polynomial.sav'

    # classification
    elif model_no in [3, 4, 5]:
        data = merged = merge_samples_directors('./Milestone 2/movies-revenue-test-samples.csv',
                                                './Milestone 2/movie-director-test-samples.csv')
        if model_no == 3:
            filename = './model3_knn.sav'
        elif model_no == 4:
            filename = './model4_randomForest.sav'
        else:
            filename = 'model5_svm.sav'

    model = joblib.load(filename)

    if model_no == 1:
        fetched = fetched_ord_cols
        X_test, Y_test = setting_xy_for_predict(data, 'revenue', fetched)
    elif model_no == 2:
        fetched = fetched_dumm
        X_test, Y_test = settingXandYUsingDummies(data, fetched)
    elif model_no == 3 or model_no == 5:
        fetched = fetched_ord_cols
        X_test, Y_test = setting_xy_for_classefiers(data, 'MovieSuccessLevel', fetched)
    elif model_no == 4:
        fetched = fetched_dumm
        X_test, Y_test = setting_xy_for_random(data, 'MovieSuccessLevel', fetched)

    # data = pd.read_csv('testComplete.csv')
    pred = model.predict(X_test)
    print('Mean Square Error  => ', metrics.mean_squared_error(Y_test, pred))
    print('r2:', metrics.r2_score(Y_test, pred))


main(4)


