import preprocessing
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../[MERGED-COMPLETE]movies_revenue2.csv')


def train_model(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, shuffle=True, random_state=10)
    start = time.time()
    ridge = Ridge(alpha=1.0, solver='auto')
    ridge.fit(X_train, y_train)
    joblib.dump(ridge, '../saved_models/model1_ridge.sav')
    end = time.time()
    print(X.shape)
    y_pred = ridge.predict(X_test)
    print('Mean Square Error ridge test => ',
          metrics.mean_squared_error(y_pred, y_test))
    print('r2 score Test ridge=> ', metrics.r2_score(y_test, y_pred))
    y_pred = ridge.predict(X_train)
    print('Mean Square Error ridge train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train ridge => ', metrics.r2_score(y_train, y_pred))
    print("time of training => ", end-start)


def __main__():
    X, Y = preprocessing.setting_xy_for_predict(data, 'revenue')
    train_model(X, Y)


__main__()
