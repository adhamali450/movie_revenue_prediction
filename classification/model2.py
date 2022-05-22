from sklearn.multiclass import OneVsOneClassifier
from sklearn import metrics
from sklearn.svm import SVC
from preprocessing import *
from sklearn.model_selection import train_test_split
import time

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue.csv')

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=84, shuffle=True)
    svm = SVC(C=100, degree=3, gamma=0.1, kernel='rbf')
    ovo = OneVsOneClassifier(svm)

    train_start = time.time()
    ovo = ovo.fit(X_train, y_train)
    train_end = time.time()

    test_start = time.time()
    y_pred = ovo.predict(X_test)
    test_end = time.time()

    train_time = train_end-train_start
    test_time = test_end-test_start
    accuracy = metrics.accuracy_score(y_test, y_pred)

    print('Mean Square Error OVO SVM test => ',
          metrics.mean_squared_error(y_pred, y_test))
    print('r2 score Test OVO SVM=> ', metrics.accuracy_score(y_test, y_pred))
    y_pred = ovo.predict(X_train)
    print('Mean Square Error OVO SVM train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train OVO SVM => ', metrics.accuracy_score(y_train, y_pred))

    plot_numbers = [accuracy, train_time, test_time]
    graph_bar(plot_numbers, 'Time', 'SVM One-vs-One Model')

def __main__():
    X, Y = setting_xy_for_SVM(data, 'MovieSuccessLevel')
    train_model(X, Y)

__main__()
