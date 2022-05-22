from preprocessing import *
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue.csv')

def train_random_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, shuffle=True, random_state=84)
    rf = RandomForestRegressor(n_estimators=1000)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print('Mean Square Error RandomForestRegressor test => ',
          metrics.mean_squared_error(y_test, y_pred))
    print('r2 score Test RandomForestRegressor => ',
          metrics.r2_score(y_test, y_pred))
    y_pred = rf.predict(X_train)
    print('Mean Square Error RandomForestRegressor train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train RandomForestRegressor => ',
          metrics.r2_score(y_train, y_pred))

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, shuffle=True, random_state=10)

    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                    n_estimators=1000, learning_rate=0.01)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    print('Mean Square Error DecisionTreeClassifier test => ',
          metrics.mean_squared_error(y_test, prediction))
    print('r2 score Test DecisionTreeClassifier=> ',
          metrics.r2_score(y_test, prediction))
    prediction = model.predict(X_train)
    print('Mean Square Error DecisionTreeClassifier train =>',
          metrics.mean_squared_error(y_train, prediction))
    print('r2 score Train DecisionTreeClassifier => ',
          metrics.r2_score(y_train, prediction))

def __main__():
    #X, Y = setting_xy_for_knn(data, 'MovieSuccessLevel')
    #train_model(X, Y)
    X, Y = setting_xy_for_random(data, 'MovieSuccessLevel')
    train_random_model(X, Y)

__main__()
