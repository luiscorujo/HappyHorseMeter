import cv2 as cv
import numpy as np
import os



path = '/Users/luis/Desktop/data_wider_not/val/Relaxed/'
dir = os.listdir(path)

for image in dir:
    if image.endswith('.jpeg') or image.endswith('.jpg') or image.endswith('.png') or image.endswith('.JPEG'):
        img = cv.imread(path+image)
        print(img)
        resized_image = cv.resize(img, (150, 150))
        cv.imwrite('/Users/luis/Desktop/data_wider/val/Relaxed/'+image, resized_image)
