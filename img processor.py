# import numpy as np
# import cv2 as cv
# import os
# img = cv.imread('/5011 ss2/5011 ss0001.tif', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# ret,thresh = cv.threshold(img,127,255,0)
# contours,hierarchy = cv.findContours(thresh, 1, 2)

# cnt = contours[0]
# M = cv.moments(cnt)
# print( M )

import os
import cv2 as cv

# Check if the file exists
file_path = 'C:/Users/Downloads/btp/5011 ss2/5011 ss0001.tif'
print(f"File exists: {os.path.exists(file_path)}")
print(f"Absolute path: {os.path.abspath(file_path)}")