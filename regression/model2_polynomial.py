from sklearn import linear_model
from sklearn import metrics
import time
from sklearn.model_selection import train_test_split
import preprocessing
import pandas as pd
import joblib


data = pd.read_csv('../[MERGED-COMPLETE]movies_revenue2.csv')

def settingXandY(YColumn):
      X ,Y = preprocessing.settingXandYUsingDummies(data)
      create_model(X ,Y)

def create_model(X ,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

    poly_model = linear_model.LinearRegression()
    start = time.time()
    poly_model.fit(X_train, y_train)
    joblib.dump(poly_model, '../saved_models/model2_polynomial.sav')
    end = time.time()

    prediction = poly_model.predict(X_test)
    prediction2 = poly_model.predict(X_train)
    print('r2 score Test => ', metrics.r2_score(y_test, prediction))
    print('Mean Square Error Linear test => ', metrics.mean_squared_error(y_test, prediction))
    print('r2 score Train => ', metrics.r2_score(y_train, prediction2))
    print('Mean Square Error Linear train => ', metrics.mean_squared_error(y_train, prediction2))
    preprocessing.plot_graph(y_test, prediction, "Linear Model")
    print("time of training => " ,end-start)

def __main__():
    settingXandY('revenue')
__main__()