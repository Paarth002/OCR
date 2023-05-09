import cv2
import numpy as np

img = cv2.imread('static/classify.jpg')
print(type(img), img.shape)
print (img[:2, :2, :])