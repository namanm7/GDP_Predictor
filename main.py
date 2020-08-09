import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer

def fillMissing(file, cols):
    GDPData = pd.read_csv(file)
    imputer = KNNImputer(n_neighbors=2)
    df_filled = imputer.fit_transform(GDPData)
    df=pd.DataFrame(df_filled, columns = cols)
    return df

def doReg(df, GDPName, Reg):
    x = df.loc[:, df.columns != GDPName]
    y = df[[GDPName]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state=5)

    model = Reg.fit(x_train, y_train)
    ypred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    print("R2:{0:.3f}".format(r2))
    return model, x_test, y_test, ypred


def printPrediction(Q2, model):
    predict = model.predict(Q2)
    print("Predicted GDP: %.2f percent" %predict)
    anualized = predict * 4
    print("Annualized: %.2f" %anualized)

def plotPrediction(x_test, y_test, ypred):
    x_ax = range(len(x_test))
    plt.scatter(x_ax, y_test, s=5)
    plt.plot(x_ax, ypred, lw=0.8)
    plt.show()

def runMain(file, cols, Q2, GDPName):
    df = fillMissing(file, cols)

    alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    elastic_cv=ElasticNetCV(alphas=alphas)

    model, x_test, y_test, ypred = doReg(df, GDPName, elastic_cv)
    printPrediction(Q2, model)
    plotPrediction(x_test, y_test, ypred)

    ridge_cv = RidgeCV()
    model, x_test, y_test, ypred = doReg(df, GDPName, ridge_cv)
    printPrediction(Q2, model)
    plotPrediction(x_test, y_test, ypred)
'''
file = "Bare_GDPData.csv"
cols = ['GDPC1','CCSA','UMCSENT','UNRATE','CPIAUCSL']
Q2 = [[763.3,74.1,240.0,-0.9]]
GDPName = 'GDPC1'

'''

file = "8-7-GDPData.csv"
cols = ['GDPC1','CCSA','NEWORDER','UMCSENT','PAYEMS','CPIAUCSL','MSACSR']
Q2 = [[19564308,-5.5,74.1,-12.0,-0.9,5.7]]
GDPName = 'GDPC1'

runMain(file, cols, Q2, GDPName)



#GDPData = pd.read_csv("NEW_GDPData.csv")
'''
GDPData = pd.read_csv("Bare_GDPData.csv")
#cols = ['Unemp_claims','Man_New_Order','GDP','CPI','Labor_Force','Consumer_Sentiment','Monthly_House_Supply','Mortgage_Debt']
cols = ['GDPC1','CCSA','UMCSENT','UNRATE','CPIAUCSL']

x = df[['CCSA','UMCSENT','UNRATE','CPIAUCSL']]
y = df[['GDPC1']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state=5)


alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(x_train, y_train)
ypred = model.predict(x_test)
score = model.score(x_test, y_test)
mse = mean_squared_error(y_test, ypred)
print("R2:{0:.4f}, MSE:{1:.4f}, RMSE:{2:.4f}".format(score, mse, np.sqrt(mse)))

x_ax = range(len(x_test))
plt.scatter(x_ax, y_test, s=5)
plt.plot(x_ax, ypred, lw=0.8)
plt.legend()
plt.show()

Q2 = [[763.3,74.1,240.0,-0.9]]

predict = model.predict(Q2)
print(type(predict))
print("Predicted GDP: %.2f percent" %predict)
'''

#x = df[['Unemp_claims','Man_New_Order','CPI','Labor_Force','Consumer_Sentiment','Monthly_House_Supply','Mortgage_Debt']]


#mlr = LinearRegression()

#model=mlr.fit(x_train, y_train)

#y_predict = mlr.predict(x_test)
#print("Train score:")
#print(mlr.score(x_train, y_train))

#print("Testing score:")
#print(mlr.score(x_test,y_test))

#Q2 = [[19564308,189079,256.295,158213,74.1,5.7,4.0]]


#plt.scatter(y_test, y_predict)
#plt.plot(range(20000), range(20000))

#plt.xlabel("GDP: $Y_i$")
#plt.ylabel("Predicted GDP: $\hat{Y}_i$")
#plt.title("Actual GDP vs Predicted GDP")

#plt.show()
