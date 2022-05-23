import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from preprocessing import *
import warnings
import joblib

warnings.filterwarnings("ignore")

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue2.csv')

def applying_GridSearch(X_train, y_train):
    grid_params = {'n_neighbors': [5, 7, 13, 15],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski', 'euclidean', 'manhattan']}
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=4, refit='r2')
    g_res = gs.fit(X_train, y_train)
    print(g_res.best_params_)
    print(g_res.best_estimator_)

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=25, shuffle=True)
    knn = KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='distance')

    train_start = time.time()
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'model1.sav')
    train_end = time.time()

    test_start = time.time()
    y_pred = knn.predict(X_test)
    test_end = time.time()

    accuracy = metrics.accuracy_score(y_test, y_pred)
    train_time = train_end - train_start
    test_time = test_end - test_start

    print('Mean Square Error KNN test => ',
          metrics.mean_squared_error(y_pred, y_test))
    print('r2 score Test KNN=> ', metrics.accuracy_score(y_test, y_pred))
    y_pred = knn.predict(X_train)
    print('Mean Square Error KNN train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train KNN => ', metrics.accuracy_score(y_train, y_pred))

    plot_numbers = [accuracy, train_time, test_time]
    graph_bar(plot_numbers, 'Time', 'KNN Model')

def __main__():
    X, Y = setting_xy_for_classifiers(data, 'MovieSuccessLevel')
    train_model(X, Y)

__main__()