import numpy as np
from PIL import Image
import pandas
import matplotlib.pyplot as plt
import os
import csv

data_path="INSERT PATH TO FOLDER OF IMAGES"

"""Returns the average red pixel value of a jpeg"""
def Redify(path_to_image):
    image = Image.open(path_to_image).convert('RGB')  # Ensure it's in RGB format
    imagey = np.array(image)
    red_image_vals = imagey[:,:,0]
    num_rows, num_columns = red_image_vals.shape
    redvalue = np.sum(red_image_vals)
    return redvalue/(num_rows*num_columns)



"""Returns a csv with chemical values, and red values of image"""
def process_image(input_folder):
    image_files=[name for name in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder,name))]
    with open('test_data.csv','w', newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Name','Red Value'])
        i=0
        for image_name in image_files:
            i+=1 
            image_path = os.path.join(input_folder, image_name)
            image_array = np.array(Image.open(image_path))
            red_value = Redify2(image_array)
            writer.writerow([i,red_value])
        
    return



if __name__== "__main__":
    print("starting")
    process_image("data_path")
    print("Image processing completed successfully")
