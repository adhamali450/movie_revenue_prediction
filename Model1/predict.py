import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

    rdg = Ridge(alpha = 0.01 , normalize = True)
    rdg.fit(X, Y)
    prediction = rdg.predict(X_test)
    print('Mean Square Error ridge test => ', metrics.mean_squared_error(y_test, prediction))
    print('R2 => ', metrics.r2_score(y_test, prediction))
    prediction = rdg.predict(X_train)
    print('Mean Square Error ridge train =>', metrics.mean_squared_error(y_train, prediction))
    print('R2 => ', metrics.r2_score(y_train, prediction))

def __main__():
    X, Y = preprocessing.settingXandY('revenue')
    train_model(X, Y)
__main__()
