import numpy as np
from PIL import Image
import pandas
import matplotlib.pyplot as plt
import os
import csv
# import imageio as iio


""" This file is expecting images to be JPEG's."""


"""Downloaded text image set from Kaggle at: https://www.kaggle.com/datasets/pavansanagapati/images-dataset"""

# test_img=iio.v2.imread("data/test_photos/red_plant.jpeg",pilmode='RGB')

test_im=np.array(Image.open("data/test_photos/red_plant.jpeg"))

# print(red_values)


"""Returns the average red pixel value of a jpeg"""
def Redify(imagey):

    red_image_vals=imagey[:,:,0]
    num_rows, num_columns = red_image_vals.shape
    redvalue=0
    for i in range(num_rows):
        for j in range(num_columns):
            redvalue+=red_image_vals[i][j]
    # print(redvalue/(num_rows*num_columns))
    return(redvalue/(num_rows*num_columns))
# 15min+2:30
def Redify2(imagey):
    red_image_vals=imagey[:,:,0]
    num_rows, num_columns = red_image_vals.shape
    redvalue=np.sum(red_image_vals)
    return redvalue/(num_rows*num_columns)




def process_image(input_folder):
    image_files=[name for name in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder,name))]
    with open('test_data.csv','w', newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Name','Red Value'])
        for image_name in image_files:
            image_path = os.path.join(input_folder,image_name)
            image_array = np.array(Image.open(image_path))
            red_value = Redify2(image_array)
            writer.writerow([image_name,red_value])
        
    return

process_image("/Users/kadenparker/Desktop/Misc/BYU-IGEM-2024/src/data/test_photos/data/cars")

Redify(test_im)



#2:42-2:54

#APP should automatically change in settings the camera capture in IOS from high efficiency to most compatible



#Types of iamges O m expectting


if __name__=="__main__":
    print("starting")
    process_image("/Users/kadenparker/Desktop/Misc/BYU-IGEM-2024/src/data/test_photos/data/flowers")
    print("Image processing completed successfully")