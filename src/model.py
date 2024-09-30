import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib


file = "test_data.csv" #Path to data csv, will be in the form of amount of chemical, and red pixel values of image
data = pd.read_csv(file) #read in data
X = data[["Name"]]


Y = data["Red Value"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
joblib.dump(model, 'AI_chemical_in_soil_predicting_model.joblib')


#testing model
red_value_to_predict = 10
predicted_chemical_amount = model.predict([[red_value_to_predict]])
print(f"Predicted chemical amount for red value {red_value_to_predict}: {predicted_chemical_amount[0]}")


if __name__== "__main__":
   pass
