import re
import datetime
from datetime import timedelta
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder , OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def movie_lemma(movie):
    return re.sub("\d*", "", movie).strip()

def get_sequels(data):
    movies = data['movie_title'].values.tolist()
    is_sequels = [0 for movie in movies]

    for i in range(len(movies)):
        for j in range(i + 1, len(movies)):
            if movie_lemma(movies[i]) == movie_lemma(movies[j]):
                is_sequels[j] = 1

    data.insert(1, 'sequel', is_sequels)

def merge_files(fileOnePath, fileTwoPath, commonColumn):
    df1 = pd.read_csv(fileOnePath)
    df2 = pd.read_csv(fileTwoPath)
    
    df2 = df2.rename(columns={'name': 'movie_title'})

    df = pd.merge(df1, df2, on=commonColumn, how='outer')
    return df

def feature_encoder(data, nom_cols, ord_cols):
    encoder = ce.OneHotEncoder(
        cols=ord_cols, handle_unknown='return_nan', return_df=True, use_cat_names=True)
    temp_data = encoder.fit_transform(data)
    
    cols = data.columns
    cols = cols.drop(ord_cols)
    temp_data.drop(columns= cols, axis=1, inplace=True)
    
    with open('../ord_cols.txt', 'w') as f:
        for col in temp_data.columns:
            f.write(col + '\n')

    data = encoder.fit_transform(data)

    for c in nom_cols:
        frequency_encoding(data, c)

    return data

def drop_contextual_na(data):
    # actual nulls and rows that evaluates to null
    # e.g. director: Unknown
    data.dropna(how='any', inplace=True)

    data.drop(data.index[data['director'] == 'Unknown'], inplace=True)
    data.drop(data.index[data['director'] == 'NO_DIRECTOR'], inplace=True)
    data.drop(data.index[data['MPAA_rating'] == 'UNKNOWN'], inplace=True)
    data.drop(data.index[data['MPAA_rating'] == 'Not Rated'], inplace=True)
    data.drop('animated', axis=1, inplace=True)

def unformat_revenue(data):
    data['revenue'] = data['revenue'].replace(
        '[\$,]', '', regex=True).astype(float)

def spread_date(df, col_name, keep_original=False):
    df[col_name] = pd.to_datetime(df[col_name])
    df['year'] = df[col_name].dt.year
    df['month'] = df[col_name].dt.month
    df['day'] = df[col_name].dt.day

    new_year = []
    for year in df['year']:
        if year > 2016:
            year = 1900 + (year % 100)
            new_year.append(year)
        else:
            new_year.append(year)
    df['year'] = new_year

    if not keep_original:
        df.drop(['release_date'], axis=1, inplace=True)

def shift_target_column(data, col_name):
    move = data.pop(col_name)
    data.insert(int(data.iloc[0].size), col_name, move)

def draw_heat_map(YColumn, readyData):
    movie_data = readyData.iloc[:, :]
    corr = movie_data.corr()
    top_feature = corr.index[abs(corr[YColumn]) > 0.09]
    plt.subplots(figsize=(12, 8))
    top_corr = movie_data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    return top_feature

def plot_graph(y_test, y_pred, regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return

def prepare_data(data, target_col, work_with_revenue=True):
    # work with revenue set to False when called to a classification model

    get_sequels(data)
    drop_contextual_na(data)
    if work_with_revenue:
        unformat_revenue(data)
    spread_date(data, 'release_date', False)
    shift_target_column(data, target_col)

def corr_feature_extraction(YColumn,readyData):
    movie_data = readyData.iloc[:, :]
    corr = movie_data.corr()
    top_feature = corr.index[abs(corr[YColumn]) > 0.09]
    top_corr = movie_data[top_feature].corr()
    top_feature = top_feature.delete(-1)
    return readyData[top_feature]


def lasso_feature_selection(X, Y):
    lasso = Lasso().fit(X, Y)
    model = SelectFromModel(lasso, prefit=True)
    return model.transform(X)

def setting_xy_for_classifiers(data, target_col):
    prepare_data(data, target_col, False)

    # encoding
    ## 1. features
    data_encoded = feature_encoder(data, ['director'], ['genre', 'MPAA_rating'])
    ## 2. label
    encoder = LabelEncoder()
    data_encoded[target_col] = encoder.fit_transform(data_encoded[target_col])

    X = data_encoded.iloc[:, 0:]  # Features
    X.drop(['movie_title', target_col], axis=1, inplace=True)
    print(X.columns)
    Y = data_encoded[target_col]  # Label

    X_new = lasso_feature_selection(X, Y)

    # scaling
    scale = StandardScaler()
    X_new = scale.fit_transform(X_new)

    return X_new, Y

def setting_xy_for_predict(data, target_col):
    prepare_data(data, 'revenue')

    # encoding
    data_encoded = feature_encoder(data, ['movie_title','director'], ['genre', 'MPAA_rating'])

    # corr graph
    draw_heat_map(target_col, data_encoded)

    # discarding outliers
    min_threshold, max_threshold = data_encoded[target_col].quantile([0.01, 0.99])
    data_ready = data_encoded[(data_encoded[target_col] > min_threshold) & (
            data_encoded['revenue'] < max_threshold)]

    X = data_ready.iloc[:, 0:]  # Features

    Y = data_ready[target_col]  # Label
    X.drop('revenue', axis=1, inplace=True)
    print(X.columns)

    # feature Selection
    X_new = lasso_feature_selection(X, Y)
    # X_new = X

    # scaling
    scale = StandardScaler()
    X_new = scale.fit_transform(X_new)

    return X_new, Y

def settingXandYUsingDummies(data):
    prepare_data(data, 'revenue')

    colName = ['genre', 'MPAA_rating']
    dummeis = pd.get_dummies(data[colName], prefix_sep="_", drop_first=True)
    merge_data = pd.concat([data, dummeis], axis=1)
    merge_data.drop(columns=colName, axis=1, inplace=True)
    shift_target_column(merge_data, 'revenue')

    with open('../dumm.txt', 'w') as f:
        for col in dummeis.columns:
            f.write(col + '\n')

    cols = ['movie_title', 'director']
    for c in cols:
        frequency_encoding(merge_data, c)

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

    return X , Y

def frequency_encoding(data, col_name):
    dic = {col_name: (data[col_name].value_counts() /
                      len(data[col_name])).to_dict()}
    data[col_name] = data[col_name].map(dic[col_name])


def setting_xy_for_random(data, target_col):
    prepare_data(data, target_col, False)

    #Generate Dummies for Genres and MPAA_ratings
    colName = ['genre', 'MPAA_rating']
    dummeis = pd.get_dummies(data[colName], prefix_sep="_", drop_first=False)
    merge_data = pd.concat([data, dummeis], axis=1)
    merge_data.drop(columns=colName, axis=1, inplace=True)
    shift_target_column(merge_data, target_col)

    with open('../dumm.txt', 'w') as f:
        for col in dummeis.columns:
            f.write(col + '\n')

    cols = ['director']
    for c in cols:
        frequency_encoding(merge_data, c)

    merge_data.drop(['movie_title'], inplace=True, axis=1)
    X = merge_data.iloc[:, 0:]
    Y = merge_data.iloc[:, -1:]
    X.drop(['MovieSuccessLevel'], axis=1, inplace=True)
    print(X.shape)
    
    levelOfSuccess = LabelEncoder()
    Y = levelOfSuccess.fit_transform(Y)

    return X, Y

def graph_bar(plot_numbers, y_label, plot_title):
    plot_data = ['Accuracy', 'Training Time', 'Test Time']
    plt.bar(plot_data, plot_numbers)
    plt.title(plot_title)
    plt.ylabel(y_label)
    plt.show()
