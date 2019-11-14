import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras import layers
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

my_model = XGBRegressor()


a=pd.read_csv("train.csv")
cechy=["a","b"]

print(a[cechy].head())

zm1=a.a.values
zm2=a.b.values
zm3=a.znak.values

X=a[cechy].values
y=a.wynik.values
X.shape

model = Sequential()
model.add(layers.Dense(1))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])







my_model.fit(X,y)

b=pd.read_csv("train.csv")
print(b[cechy].head())
X_test=b[cechy].values



preds=my_model.predict(X_test)
print(preds[0:10])
c=pd.read_csv("wyniki.csv")
y_wyniki=c.wynik.values
print(y_wyniki[0:10])
print("Mean Absolute Error: " + str(mean_absolute_error(preds, y_wyniki)))
