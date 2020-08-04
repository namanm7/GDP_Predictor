import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

GDPData = pd.read_csv("NEW_GDPData.csv")
cols = ['Unemp_claims','Man_New_Order','GDP','CPI','Labor_Force','Consumer_Sentiment','Monthly_House_Supply','Mortgage_Debt']
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df_filled = imputer.fit_transform(GDPData)
df=pd.DataFrame(df_filled, columns = cols)


#df = pd.DataFrame(GDPData)
print(df)

x = df[['Unemp_claims','Man_New_Order','CPI','Labor_Force','Consumer_Sentiment','Monthly_House_Supply','Mortgage_Debt']]

y = df[['GDP']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)
print("Train score:")
print(mlr.score(x_train, y_train))

print("Testing score:")
print(mlr.score(x_test,y_test))

Q2 = [[19564308,189079,256.295,158213,74.1,5.7,100000]]

predict = mlr.predict(Q2)

print("Predicted GDP: ", predict)

plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))

plt.xlabel("GDP: $Y_i$")
plt.ylabel("Predicted GDP: $\hat{Y}_i$")
plt.title("Actual GDP vs Predicted GDP")

plt.show()
