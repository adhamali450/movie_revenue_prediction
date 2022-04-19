from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('C:\\Users\\Sondos\\Documents\\GitHub\\movie_revenue_prediction\\datasets\\[MERGED-COMPLETE]movies_revenue.csv')

def droppingNa():
    data.dropna(how='any', inplace=True)

def dataDroping():
    data.drop(data.index[data['directors'] == 'Unknown'], inplace=True)
    data.drop(['release_date'], axis=1, inplace=True)

def currency():
    data['revenue'] = data['revenue'].replace('[\$,]', '', regex=True).astype(float)

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
    movingRevenueColumn(data,'revenue')

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
    dummeis = pd.get_dummies(data[['genre', 'MPAA_rating']], prefix_sep="_", drop_first=True)
    merge_data = pd.concat([data, dummeis], axis=1)
    merge_data.drop(['genre', 'MPAA_rating'], axis=1, inplace=True)
    lbl = LabelEncoder()
    merge_data['directors'] = lbl.fit_transform(merge_data['directors'])
    movingRevenueColumn(merge_data,YColumn)
    creatModel(merge_data)

def creatModel(merge_data):
    X = merge_data.drop(['revenue', 'movie_title', 'animated'], axis=1)
    Y = merge_data['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train, y_train)

    prediction = poly_model.predict(X_test)
    prediction2 = poly_model.predict(X_train)
    print(metrics.r2_score(y_test, prediction))
    print('Mean Square Error Linear test => ', metrics.mean_squared_error(y_test, prediction))
    print(metrics.r2_score(y_train, prediction2))
    print('Mean Square Error Linear train => ', metrics.mean_squared_error(y_train, prediction2))

def __main__():
    settingXandY('revenue')
__main__()