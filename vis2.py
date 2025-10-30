import cv2
import numpy as np
from conot import get_outer_contour
from conot import contour_area
from conot import contour_circularity
from conot import contour_smoothness
from conot import visualize_contour
import os

path = "C:/Users/Aaryan/Downloads/btp/5011 ss2/5011 ss0098.tif"
print("Exists:", os.path.exists(path))


# Load the image

contour = get_outer_contour(path)
print("Area:", contour_area(contour))
print("Circularity:", contour_circularity(contour))
print("Smoothness:", contour_smoothness(contour))
visualize_contour(path,contour)