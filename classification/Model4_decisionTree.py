from preprocessing import *
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue.csv')

def applying_GridSearch(X_train, y_train):
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3, 4],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

def train_random_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, shuffle=True, random_state=84)

    rf = RandomForestRegressor(bootstrap=True, max_depth=110, max_features=4, min_samples_leaf=3, min_samples_split=10, n_estimators=100, random_state=85)
    # Train the model on training data
    train_start = time.time()
    rf.fit(X_train, y_train)
    train_end = time.time()

    test_start = time.time()
    y_pred = rf.predict(X_test)
    test_end = time.time()

    accuracy = metrics.r2_score(y_test, y_pred)
    train_time = train_end-train_start
    test_time = test_end-test_start

    print('Mean Square Error RandomForestRegressor test => ',
          metrics.mean_squared_error(y_test, y_pred))
    print('r2 score Test RandomForestRegressor => ',
          metrics.r2_score(y_test, y_pred))
    y_pred = rf.predict(X_train)
    print('Mean Square Error RandomForestRegressor train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train RandomForestRegressor => ',
          metrics.r2_score(y_train, y_pred))

    plot_numbers = [accuracy, train_time, test_time]
    graph_bar(plot_numbers, 'Time', 'Random Forest Model')

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
