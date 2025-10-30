import cv2
import numpy as np

def get_outer_contour(img_path, draw_color=(255, 0, 0), thickness=3):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    
    return largest_contour

def contour_area(contour):
    """Return area of the contour."""
    return cv2.contourArea(contour)

def contour_circularity(contour):
    """Return circularity of the contour (1 = perfect circle)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def contour_smoothness(contour):
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area == 0:
        return 0
    return (perimeter ** 2) / (4 * np.pi * area)


# Example usage (assuming get_outer_contour is defined):
# contour = get_outer_contour("your_image.png")
# print("Area:", contour_area(contour))
# print("Circularity:", contour_circularity(contour))
# print("Smoothness:", contour_smoothness(contour))


def visualize_contour(img_path, contour, draw_color=(255, 0, 0), thickness=3, window_name="Contour"):
    """
    Draws the given contour on the image and displays it.

    Parameters:
        img_path (str): Path to the input image.
        contour (numpy.ndarray): Contour points (from get_outer_contour).
        draw_color (tuple): BGR color for drawing (default: blue).
        thickness (int): Thickness of contour line.
        window_name (str): Name of the display window.
    """
    # Load the image
    img = cv2.imread(img_path)

    # Draw the contour
    contour_img = img.copy()
    cv2.drawContours(contour_img, [contour], -1, draw_color, thickness)

    # Show the result
    cv2.imshow(window_name, contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contour_img
