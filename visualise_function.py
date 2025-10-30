import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import cv2 as cv
import os
from skimage.morphology import medial_axis
from glob import glob
from PIL import Image

def contno(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    centroids = []
    thre = 10
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= thre:
            centroids.append(cnt)
    return centroids[:-1]

def conotur_centroid(image_path):
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
    return centroids[:-1]

def predict_displacement(centers_t, centers_t1):
    centers_t = np.array(centers_t)
    centers_t1 = np.array(centers_t1)
    disp_vectors = []
    for i, c in enumerate(centers_t):
        dists = np.linalg.norm(centers_t1 - c, axis=1)
        min_idx = np.argmin(dists)
        disp_vectors.append(centers_t1[min_idx] - c)
    return disp_vectors

def find_contours_inside_mask(image_path, mask_contour):
    contours = contno(image_path)
    centroids = conotur_centroid(image_path)
    mask = np.zeros_like(cv.imread(image_path, cv.IMREAD_GRAYSCALE), dtype=np.uint8)
    cv.drawContours(mask, [mask_contour], -1, 255, -1)
    inside_contours_idx = []
    for idx, centroid in enumerate(centroids):
        x, y = centroid
        if mask[int(y), int(x)] > 0:
            inside_contours_idx.append(idx)
    return inside_contours_idx

def track_all_descendants(image_paths, sel_frame_idx, sel_contour_idx):
    # Start with selected contour in selected frame
    contours = [contno(p) for p in image_paths]
    centroids = [conotur_centroid(p) for p in image_paths]
    out_tracks = []
    mask_contour = contours[sel_frame_idx][sel_contour_idx]
    for i in range(sel_frame_idx, len(image_paths)-1):
        idxs_in_next = find_contours_inside_mask(
            image_paths[i+1],
            mask_contour)
        out_tracks.append((i+1, idxs_in_next))
        # Optional: update mask_contour, e.g. average of descendants, or just repeat last mask
    return out_tracks

def make_gif_all(image_paths, descendant_tracks, outname="tracked_cells.gif"):
    gif_frames = []
    for frame_idx, contours_idx_list in descendant_tracks:
        img = cv.imread(image_paths[frame_idx])
        display_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        contours = contno(image_paths[frame_idx])
        # Draw all contours
        cv.drawContours(display_img, contours, -1, (180,180,180), 1)
        # Highlight selected contours
        for idx in contours_idx_list:
            cv.drawContours(display_img, [contours[idx]], -1, (0,255,0), 4)
        gif_frames.append(Image.fromarray(display_img))
    gif_frames[0].save(outname, save_all=True, append_images=gif_frames[1:], duration=300, loop=0)


def visualise(folder_path):
    image_paths = sorted(glob(os.path.join(folder_path, "*")))
    frame_idx = 0

    # Step 1: Show frame selector
    fig, ax = plt.subplots()
    curr_img = cv.imread(image_paths[frame_idx])
    img_rgb = cv.cvtColor(curr_img, cv.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(f"Select a frame (current: {frame_idx + 1}/{len(image_paths)})")

    def next_frame(event):
        nonlocal frame_idx
        frame_idx = (frame_idx + 1) % len(image_paths)
        img = cv.imread(image_paths[frame_idx])
        ax.clear()
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(f"Select a frame (current: {frame_idx + 1}/{len(image_paths)})")
        plt.draw()

    def prev_frame(event):
        nonlocal frame_idx
        frame_idx = (frame_idx - 1) % len(image_paths)
        img = cv.imread(image_paths[frame_idx])
        ax.clear()
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(f"Select a frame (current: {frame_idx + 1}/{len(image_paths)})")
        plt.draw()

    axprev = plt.axes([0.7, 0.02, 0.12, 0.05])
    axnext = plt.axes([0.82, 0.02, 0.12, 0.05])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next_frame)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(prev_frame)
    plt.show(block=False)

    print("Use matplotlib buttons or click 'Next'/'Previous' to select your frame, then close the figure window.")

    # Wait for frame selection, then proceed
    plt.show()

    # Step 2: Show contours, let user select one
    contours = contno(image_paths[frame_idx])
    fig2, ax2 = plt.subplots()
    img_disp = cv.imread(image_paths[frame_idx])
    img_rgb_disp = cv.cvtColor(img_disp, cv.COLOR_BGR2RGB)
    ax2.imshow(img_rgb_disp)
    for i, cnt in enumerate(contours):
        cnt = cnt.squeeze()
        x, y = cnt[:, 0], cnt[:, 1]
        ax2.plot(x, y, label=str(i))
    ax2.legend(title='Contour index')
    ax2.set_title(f"Click on a contour label to select (frame {frame_idx + 1})")

    sel_contour_idx = [None]

    def on_pick(event):
        label = event.artist.get_label()
        sel_contour_idx[0] = int(label)
        plt.close()  # Close figure after selection

    for line in ax2.get_lines():
        line.set_picker(True)
    fig2.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    if sel_contour_idx[0] is None:
        print("No contour selected.")
        return

    # Step 3: Track selected contour, make GIF
    print(f"Tracking contour {sel_contour_idx[0]} in frame {frame_idx + 1} across sequence.")
    contour_tracks = track_all_descendants(image_paths, frame_idx, sel_contour_idx[0])
    make_gif_all(image_paths, contour_tracks)

    print("Tracking GIF saved as tracked.gif in working directory.")

# Usage:
visualise('C:/Users/Aaryan/Downloads/btp/5011 ss2')
