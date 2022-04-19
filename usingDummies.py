from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('datasets/[MERGED-COMPLETE]movies_revenue.csv')

def droppingNa():
    data.dropna(how='any', inplace=True)

def dataDroping():
    data.drop(data.index[data['directors'] == 'Unknown'], inplace=True)
    data.drop(['release_date'], axis=1, inplace=True)

def currency():
    data['revenue'] = data['revenue'].str[1:].str.replace(',', '').astype("float32").astype("int32")

def sepDate(dateColumn):
    data[dateColumn] = pd.to_datetime(data[dateColumn])
    data['year'] = data[dateColumn].dt.year
    data['month'] = data[dateColumn].dt.month
    data['day'] = data[dateColumn].dt.day
    newYear = []
    for year in data['year']:
        if year > 2016:
            year = 1900 + (year % 100)
            newYear.append(year)
        else:
            newYear.append(year)
    data['year'] = newYear

def movingRevenueColumn(dataToShift,movingYColumn):
    move = dataToShift.pop(movingYColumn)
    dataToShift.insert(int(dataToShift.iloc[0].size), movingYColumn, move)

def cleaningData():
    droppingNa()
    currency()
    sepDate('release_date')
    dataDroping()
    movingRevenueColumn(data ,'revenue')

def DrawHeatMap(YColumn,readyData):
    movie_data = readyData.iloc[:, :]
    corr = movie_data.corr()
    top_feature = corr.index[abs(corr[YColumn]) > 0.09]
    plt.subplots(figsize=(12, 8))
    top_corr = movie_data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    return top_feature

def settingXandY(YColumn):
    cleaningData()
    #genre dummy
    dummeis = pd.get_dummies(data[['genre', 'MPAA_rating', 'directors', 'year', 'month']], prefix_sep="_", drop_first=True)
    merge_data = pd.concat([data, dummeis], axis=1)
    merge_data.drop(['genre', 'MPAA_rating', 'directors', 'year', 'month'], axis=1, inplace=True)
    movingRevenueColumn(merge_data,YColumn)
    print(merge_data.to_csv('dummy.csv'))
    creatModel(merge_data)

def creatModel(merge_data):
    dummy_data = pd.read_csv('dummy.csv')
    model = linear_model.LinearRegression()
    print(merge_data.columns)
    X = merge_data.drop(['revenue', 'movie_title', 'animated'], axis=1)
    Y = merge_data['revenue']
    model.fit(X,Y)
    predictions = model.predict(X)
    print(model.score(X,Y))
    print(metrics.mean_squared_error(Y, predictions))

settingXandY('revenue')