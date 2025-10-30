import os
import numpy as np
from scipy.spatial import distance_matrix
from contr import conotur_centroid
from matplotlib import pyplot as plt


def predict_displacement(centers_t, centers_t1):
    """
    Predict displacement of each cell from frame t to t+1
    using nearest-neighbor matching.
    Returns: dict[cell_index_at_t] = displacement_vector
    """
    


def link_multiplicity(centers_src, centers_dst, displacement, threshold=20):
    """
    Determine link multiplicity categories (0,1,>1) for each source cell.
    Returns: dict[cell_index] = (category, candidate_links)
    """
    


def track_cells(image_folder, threshold=20):
    """
    Perform forward + backward cell tracking with multiplicity handling.
    Returns:
        centers_per_frame: list of arrays of cell centers
        links: dict[frame][cell_idx] = linked_cell_indices_in_next_frame
    """
    # Load images
    

# import matplotlib.pyplot as plts

def visualize_tracking(centers_per_frame, links, frame_idx):
    """
    Visualize tracking results between frame_idx and frame_idx+1.

    Args:
        centers_per_frame: list of numpy arrays of shape (N,2), cell centers per frame
        links: dict[frame][cell_idx] = list of linked cells in frame+1
        frame_idx: index of the frame to visualize (shows frame and frame+1)
    """
    

