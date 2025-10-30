import numpy as np
import cv2 as cv
import os
from contr import conotur_centroid
from contr import visualize_centroids

image_path = 'C:/Users/Aaryan/Downloads/btp/5011 ss2/5011 ss0046.tif'
contours = conotur_centroid(image_path)
visualize_centroids(image_path,contours)
