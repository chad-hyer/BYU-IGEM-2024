import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

file="test_data.csv"
data=pd.read_csv(file)
X=data[["Name"]]
print(X.shape)
Y=data["Red Value"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train, Y_train)
# model.predict([[]])
print("model set")
red_value_to_predict=10
predicted_chemical_amount = model.predict([[red_value_to_predict]])
print(f"Predicted chemical amount for red value {red_value_to_predict}: {predicted_chemical_amount[0]}")


if __name__== "__main__":
   pass