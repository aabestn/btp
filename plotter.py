import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from contr import conotur_centroid


# Exponential function
def exp_growth(x, N0, k):
    return N0 * np.exp(k * x)

# Directory containing your images
directory = "C:/Users/Aaryan/Downloads/btp/5011 ss2"

# Collect counts of centroids
counts = []
frames = []

# Loop over images in directory
for idx, filename in enumerate(sorted(os.listdir(directory))):
    image_path = os.path.join(directory, filename)

    # Only process common image types
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        centroids = conotur_centroid(image_path)
        num_centroids = len(centroids)   # <-- store the count
        counts.append(num_centroids)
        frames.append(idx + 1)  # frame numbers starting from 1
        print(f"Image {idx}: {filename}, Found {num_centroids} centroids")

# Convert to numpy arrays
frames = np.array(frames, dtype=float)
counts = np.array(counts, dtype=float)

# Fit exponential model
popt, pcov = curve_fit(exp_growth, frames, counts, p0=(counts[0], 0.01))
N0_fit, k_fit = popt

# Compute doubling time
doubling_time = np.log(2) / k_fit

# Generate smooth curve for plotting fit
x_fit = np.linspace(frames.min(), frames.max(), 200)
y_fit = exp_growth(x_fit, *popt)

# Plot data and fit
plt.figure(figsize=(8,5))
plt.plot(frames, counts, 'o', label="Observed counts")
plt.plot(x_fit, y_fit, '-', label="Exponential fit")
plt.xlabel("Frame Number")
plt.ylabel("Number of Centroids (Cells)")
plt.title(f"Cell Count vs Frame Number\nDoubling time ≈ {doubling_time:.2f} frames")
plt.grid(True)
plt.legend()
plt.show()
