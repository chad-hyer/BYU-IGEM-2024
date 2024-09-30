from redify import Redify
import joblib
import warnings

warnings.filterwarnings("ignore")


path="INSERT IMAGE PATH"

#load in ai model
loaded_model = joblib.load("AI_chemical_in_soil_predicting_model.joblib")

"""Input an image, returns and prints chemical prediction based on redness of biosensor plant image"""
def run_inference(image_path):
    red_val=Redify(image_path)
    predicted_chemical_amount = loaded_model.predict([[red_val]])[0]
    print(f'Predicted Chemical Amount: {predicted_chemical_amount}')
    return predicted_chemical_amount

if __name__== "__main__":
    run_inference(path)
