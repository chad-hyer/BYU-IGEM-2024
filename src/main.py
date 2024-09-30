from redify import Redify
import joblib
import warnings
import os
import pandas as pd

warnings.filterwarnings("ignore")

folders = ['T0','T1','T2','T3','T4']
for folder in folders:
    files = os.listdir(f"D:/Downloads/Arabidopsis Training Set/{folder}")
    for file in files:
        path=f"D:/Downloads/Arabidopsis Training Set/{folder}/{file}"
        label = file.replace('.jpg','').split('-')[1]
        #load in ai model
        loaded_model = joblib.load("D:\Documents\GitHub\BYU-IGEM-2024\src\AI_chemical_in_soil_predicting_model.joblib")

        """Input an image, returns and prints chemical prediction based on redness of biosensor plant image"""
        def run_inference(image_path):
            red_val=Redify(image_path)
            predicted_chemical_amount = loaded_model.predict([[red_val]])[0]
            print(f'{file},{label},{predicted_chemical_amount}')
            return predicted_chemical_amount

        if __name__== "__main__":
            run_inference(path)
