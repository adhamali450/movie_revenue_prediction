import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from preprocessing import *

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue.csv')


def train_model(X, Y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print('Mean Square Error KNN test => ',
          metrics.mean_squared_error(y_pred, y_test))
    print('r2 score Test KNN=> ', metrics.accuracy_score(y_test, y_pred))
    y_pred = knn.predict(X_train)
    print('Mean Square Error KNN train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train KNN => ', metrics.accuracy_score(y_train, y_pred))


def __main__():
    X, Y = setting_xy_for_knn(data, 'MovieSuccessLevel')
    train_model(X, Y)


__main__()
