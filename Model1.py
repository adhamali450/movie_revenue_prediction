import preprocessing
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
import time

data = pd.read_csv('datasets/[MERGED-COMPLETE]movies_revenue.csv')

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)
    start = time.time()
    rdg = Ridge(alpha=0.01, normalize=True)
    rdg.fit(X, Y)
    end = time.time()
    prediction = rdg.predict(X_test)
    print('Mean Square Error ridge test => ', metrics.mean_squared_error(y_test, prediction))
    print('r2 score Test ridge=> ', metrics.r2_score(y_test, prediction))
    prediction = rdg.predict(X_train)
    print('Mean Square Error ridge train =>', metrics.mean_squared_error(y_train, prediction))
    print('r2 score Train ridge => ', metrics.r2_score(y_train, prediction))
    print("time of training => " ,end-start)



def __main__():
    X, Y = preprocessing.settingXandYForPredict(data, 'revenue')
    train_model(X, Y)
__main__()
