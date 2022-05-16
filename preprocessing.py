import pandas as pd
from pyparsing import col
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

def mergeFiles(fileOnePath, fileTwoPath, commonColumn):
    df1 = pd.read_csv(fileOnePath)
    df2 = pd.read_csv(fileTwoPath)
    finalDf = df1.merge(df2, on=commonColumn)
    return finalDf

def Feature_Encoder(data, nomcols, ordcols):
    encoder=ce.OneHotEncoder(cols=ordcols,handle_unknown='return_nan',return_df=True,use_cat_names=True)
    data=encoder.fit_transform(data)
    for c in nomcols:
        frequencyEncoding(data, c)
    return data

def droppingNa(data):
    data.dropna(how='any', inplace=True)

def dataDroping(data):
    data.drop(data.index[data['directors'] == 'Unknown'], inplace=True)
    data.drop(data.index[data['animated'] == 'Unknown'], inplace=True)
    data.drop(['release_date'], axis=1, inplace=True)

def currency(data):
    data['revenue'] = data['revenue'].replace('[\$,]', '', regex=True).astype(float)

def sepDate(data, dateColumn):
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

def movingRevenueColumn(data,movingYColumn):
    move = data.pop(movingYColumn)
    data.insert(int(data.iloc[0].size), movingYColumn, move)

def cleaningData(data):
    droppingNa(data)
    currency(data)
    sepDate(data, 'release_date')
    dataDroping(data)
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

def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return

def settingXandYForPredict(data, YColumn):
    cleaningData(data)
    readyData = Feature_Encoder(data, ['directors'], ['genre', 'MPAA_rating', 'animated'])
    min_threshold, max_threshold = readyData["revenue"].quantile([0.01, 0.99])
    merge_data2 = readyData[(readyData['revenue'] > min_threshold) & (readyData['revenue'] < max_threshold)]
    X = merge_data2.iloc[:, 0:]  # Features
    X.drop(['revenue','movie_title'], axis=1, inplace=True)
    Y = merge_data2[YColumn]  # Label
    # Feature Selection 
    lasso = Lasso().fit(X, Y)
    model = SelectFromModel(lasso, prefit=True)
    X_new = model.transform(X)

    scale = StandardScaler()
    X_new = scale.fit_transform(X_new)

    return X_new,Y

def settingXandYUsingDummies(data):
    cleaningData(data)
    colName = ['genre', 'MPAA_rating', 'animated']
    dummeis = pd.get_dummies(data[colName], prefix_sep="_", drop_first=True)
    merge_data = pd.concat([data, dummeis], axis=1)
    merge_data.drop(columns=colName, axis=1, inplace=True)
    movingRevenueColumn(merge_data, 'revenue')

    cols = ['movie_title', 'directors']
    for c in cols:
        frequencyEncoding(merge_data, c)

    return merge_data

def frequencyEncoding(data, colName):
    dic = {}
    dic[colName] = (data[colName].value_counts() / len(data[colName])).to_dict()
    data[colName] = data[colName].map(dic[colName])