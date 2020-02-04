import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from cloth_detection import Detect_Clothes_and_Crop
from utils import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3
from google.colab import drive


model = Load_DeepFashion2_Yolov3()

drive.mount('/content/gdrive')
root_path = '/content/gdrive/My Drive/cloth_for_training/'  #change dir to your project folder
print(os.getcwd())

for filename in os.listdir(r"/content/gdrive/My Drive/cloth_for_training/"): 
    # print(filename)   
    img_path = root_path + filename
    # print(img_path)

    # Read image
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Read_Img_2_Tensor(img_path)

    # Clothes detection and crop the image
    img_crop = Detect_Clothes_and_Crop(img_tensor, model)

   # Pretrained classifer parameters
    PEAK_COUNT_THRESHOLD = 0.02
    PEAK_VALUE_THRESHOLD = 0.01

    Save_Image(img_crop, '/content/gdrive/My Drive/aftertraining/' + filename)
