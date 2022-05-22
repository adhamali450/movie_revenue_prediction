from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC
from preprocessing import *
from sklearn.model_selection import train_test_split

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue.csv')

def train_model(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    svm = LinearSVC()

    # Make it an OvR classifier
    ovr = OneVsOneClassifier(svm)

    # Fit the data to the OvR classifier
    ovr = ovr.fit(X_train, y_train)
    y_pred = ovr.predict(X_test)
    print('Mean Square Error OVR SVM test => ',
          metrics.mean_squared_error(y_pred, y_test))
    print('r2 score Test OVR SVM=> ', metrics.accuracy_score(y_test, y_pred))
    y_pred = ovr.predict(X_train)
    print('Mean Square Error OVR SVM train =>',
          metrics.mean_squared_error(y_train, y_pred))
    print('r2 score Train OVR SVM => ', metrics.accuracy_score(y_train, y_pred))


def __main__():
    X, Y = setting_xy_for_SVM(data, 'MovieSuccessLevel')
    train_model(X, Y)

__main__()
