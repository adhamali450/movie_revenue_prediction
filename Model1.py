import preprocessing
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import time

data = pd.read_csv('[MERGED-COMPLETE]movies_revenue.csv')

def train_model(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)
    start = time.time()
    ridge = Ridge(alpha=1.0, solver='auto')
    ridge.fit(X_train, y_train)
    end = time.time()
    
    prediction = ridge.predict(X_test)
    print('Mean Square Error ridge test => ', metrics.mean_squared_error(y_test, prediction))
    print('r2 score Test ridge=> ', metrics.r2_score(y_test, prediction))
    prediction = ridge.predict(X_train)
    print('Mean Square Error ridge train =>', metrics.mean_squared_error(y_train, prediction))
    print('r2 score Train ridge => ', metrics.r2_score(y_train, prediction))
    print("time of training => " ,end-start)



def __main__():
    X, Y = preprocessing.settingXandYForPredict(data, 'revenue')
    train_model(X, Y)
__main__()
