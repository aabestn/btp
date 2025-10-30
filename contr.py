import numpy as np
import cv2 as cv
import os
from skimage.morphology import medial_axis

def conotur_centroid(image_path):
    # Read the image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret, thresh = cv.threshold(img, 127, 255, 0)
    
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    
    centroids = []
    
    thre = 10
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= thre:
            mask = np.zeros_like(img, dtype=np.uint8)
            cv.drawContours(mask, [cnt], -1, 255, -1)  
            skeleton, distance = medial_axis(mask > 0, return_distance=True)
            pts = np.column_stack(np.nonzero(skeleton))
            if pts.shape[0] == 0:
                continue  
            
            dist_matrix = np.sum((pts[:, None, :] - pts[None, :, :])**2, axis=2)
            medoid_idx = np.argmin(np.sum(dist_matrix, axis=1))
            medoid = tuple(pts[medoid_idx][::-1])  
            centroids.append(medoid)
            print("Medoid of medial axis:", medoid)
            
    return centroids[:-1]

def are(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret, thresh = cv.threshold(img, 127, 255, 0)
    
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    
    areas = 0
    n = 0
    
    thre = 10
    
    for cnt in contours[:-1]:
        area = cv.contourArea(cnt)
        if area >= thre:
            areas = areas+area 
            n = n+1
            print("Area of contour:", area)
    
    return areas

import cv2 as cv
import numpy as np

def colony_properties(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Threshold the image
    ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    
    # Find all contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros_like(img)
    
    # Draw all contours filled into the mask (treat them as one colony)
    cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)
    
    # Now find a single contour from this filled mask
    merged_contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(merged_contours) == 0:
        return None
    
    # Take the largest merged contour (the colony)
    colony = max(merged_contours, key=cv.contourArea)
    
    area = cv.contourArea(colony)
    perimeter = cv.arcLength(colony, True)
    
    # Circularity formula = 4π * Area / (Perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    return [area, perimeter, circularity]



def visualize_centroids(image_path, centroids):
    
    # Read the original color image
    img_color = cv.imread(image_path)
    
    # Draw centroids on the image
    for cx, cy in centroids:
        cv.circle(img_color, (cx, cy), 5, (0, 0, 255), -1)  # Red circle
        # cv.putText(img_color, f"({cx},{cy})", (cx+10, cy), 
        #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display the image
    cv.imshow('Image with Centroids', img_color)
    cv.waitKey(0)
    cv.destroyAllWindows()
# import os

# # List files in the directory to verify the correct filename
# directory = 'C:/Users/Aaryan/Downloads/btp/5011 ss2'
# if os.path.exists(directory):
#     files = os.listdir(directory)
#     print("Files in directory:", files)
# else:
#     print("Directory does not exist")
#     print(os.path.abspath(directory))