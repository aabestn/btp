import cv2
import numpy as np

def centroids_in_contour(contour, centroids):
    
    inside_centroids = []

    # Ensure contour is numpy array of shape (N,1,2)
    contour = np.array(contour, dtype=np.int32)
    
    for c in centroids:
        x, y = map(float, c)  # ensure float inputs
        result = cv2.pointPolygonTest(contour, (x, y), False)
        if result >= 0:  # inside or on the boundary
            inside_centroids.append((x, y))
    
    return inside_centroids
