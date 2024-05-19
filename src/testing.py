import numpy as np
from PIL import Image
import pandas
# import imageio as iio

# test_img=iio.v2.imread("data/test_photos/red_plant.jpeg",pilmode='RGB')

test_im=np.array(Image.open("data/test_photos/red_plant.jpeg"))
red_values=test_im[:,:,0]
print(red_values)
num_rows, num_columns = red_values.shape
print(test_im.shape)
print(f"Red values array dimensions: {num_rows} rows × {num_columns} columns")

# print(test_im)
# print(test_im.size)
# Image.open("data/test_photos/red_plant.jpeg").show()



#9:20

#APP should automatically change in settings the camera capture in IOS from high efficiency to most compatible



#Types of iamges O m expectting


if __name__=="__main__":
    pass
