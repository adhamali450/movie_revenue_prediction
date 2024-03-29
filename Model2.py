from sklearn import linear_model
from sklearn import metrics
import time
from sklearn.model_selection import train_test_split
import preprocessing
import pandas as pd

data = pd.read_csv('[MERGED-COMPLETE]movies_revenue.csv')

def settingXandY(YColumn):
      merge_data2 = preprocessing.settingXandYUsingDummies(data)
      creatModel(merge_data2)

def creatModel(merge_data):
    min_threshold, max_threshold = merge_data["revenue"].quantile([0.01, 0.99])
    merge_data2 = merge_data[ (merge_data['revenue']>min_threshold) & (merge_data['revenue'] < max_threshold)]
    X = merge_data2[preprocessing.DrawHeatMap('revenue', merge_data2)]
    Y = merge_data2['revenue']

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
    preprocessing.plotGraph(y_test, prediction , "Linear Model")
    print("time of training => " ,end-start)

def __main__():
    settingXandY('revenue')
__main__()