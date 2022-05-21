from sklearn import linear_model
from sklearn import metrics
import time
from sklearn.model_selection import train_test_split
import preprocessing
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('./[MERGED-COMPLETE]movies_revenue.csv')

def settingXandY(YColumn):
      merge_data2 = preprocessing.settingXandYUsingDummies(data)
      create_model(merge_data2)

def create_model(merge_data):
    min_threshold, max_threshold = merge_data["revenue"].quantile([0.01, 0.99])
    merge_data2 = merge_data[ (merge_data['revenue']>min_threshold) & (merge_data['revenue'] < max_threshold)]
    X = merge_data2.iloc[:, 0:]  # Features
    X.drop(['revenue'], axis=1, inplace=True)
    Y = merge_data2['revenue']

    lasso = Lasso().fit(X, Y)
    model = SelectFromModel(lasso, prefit=True)
    X = model.transform(X)

    #feature scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

    poly_model = linear_model.LinearRegression()
    start = time.time()
    poly_model.fit(X_train, y_train)
    end = time.time()

    prediction = poly_model.predict(X_test)
    prediction2 = poly_model.predict(X_train)
    print('r2 score Test => ', metrics.r2_score(y_test, prediction))
    print('Mean Square Error Linear test => ', metrics.mean_squared_error(y_test, prediction))
    print('r2 score Train => ', metrics.r2_score(y_train, prediction2))
    print('Mean Square Error Linear train => ', metrics.mean_squared_error(y_train, prediction2))
    preprocessing.plot_graph(y_test, prediction, "Linear Model")
    print("time of training => " ,end-start)

def __main__():
    settingXandY('revenue')
__main__()