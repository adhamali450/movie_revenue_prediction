import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('datasets/[MERGED-COMPLETE]movies_revenue.csv')

def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))

    return X

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

def movingRevenueColumn(movingYColumn):
    move = data.pop(movingYColumn)
    data.insert(int(data.iloc[0].size), movingYColumn, move)

def cleaningData():
    droppingNa()
    currency()
    sepDate('release_date')
    dataDroping()
    movingRevenueColumn('revenue')

def DrawHeatMap(YColumn,readyData):
    movie_data = readyData.iloc[:, :]
    corr = movie_data.corr()
    top_feature = corr.index[abs(corr[YColumn]) > 0.0]
    plt.subplots(figsize=(12, 8))
    top_corr = movie_data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    return top_feature

def settingXandY(YColumn):
    cleaningData()
    cols = ('movie_title', 'genre', 'MPAA_rating', 'directors', 'animated')
    readyData = Feature_Encoder(data, cols)
    X = readyData.iloc[:, 0:]  # Features
    Y = readyData[YColumn]  # Label
    top_feature = DrawHeatMap(YColumn, readyData)
    # Feature Selection
    X = X[top_feature]
    # feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X,Y
