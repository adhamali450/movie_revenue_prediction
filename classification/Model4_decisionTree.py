import preprocessing
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../[P2-MERGED-COMPLETE]movies_revenue.csv')


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
    X, Y = preprocessing.setting_xy_for_knn(data, 'MovieSuccessLevel')
    print(X.shape)
    train_model(X, Y)


__main__()
