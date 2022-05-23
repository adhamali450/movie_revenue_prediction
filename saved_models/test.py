from operator import mod

from sklearn.impute import SimpleImputer
import preprocessing
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn import metrics
import joblib
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('movies/movies-revenue-test-samples.csv')
# df2 = pd.read_csv('movies/movie-director-test-samples.csv')


preprocessing.shift_target_column(df, 'MovieSuccessLevel')



def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.r2_score(y_test, y_pred))



def main():
    
    data = pd.read_csv('testComplete.csv')

    X_test , y_test = preprocessing.setting_xy_for_SVM(data, 'MovieSuccessLevel')

    filename = 'random_forest_regressor.sav'
    model = joblib.load(filename)
    pred = model.predict(X_test)
    print('Mean Square Error  => ', metrics.mean_squared_error(y_test, pred))
    print('r2:', metrics.r2_score(y_test, pred))
    

main()


