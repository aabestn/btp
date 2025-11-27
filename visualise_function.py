import os, zipfile, numpy as np, cv2, imageio
from PIL import Image, ImageDraw
from skimage import morphology, measure, segmentation, exposure
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
import threading, time, math

import os, zipfile
import numpy as np
from PIL import Image, ImageDraw
import imageio
import cv2
from skimage import morphology, measure, segmentation, exposure
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

# --- CONFIGURATION ---
ZIP_PATH = "./5011ss2.zip"
IMAGE_FOLDER = "./5011ss2/"
OUTPUT_FOLDER = "./5011ss2_output_post_split_fixed"
MIN_AREA = 30
USER_REQUESTED_NAME = "0"
MAX_MATCH_DIST = 150
IOU_MERGE_THRESHOLD = 0.05  

EROSION_ITERATIONS = 2
SEED_SNAP_RADIUS = 20
TRAIN_FRAMES = 5
PRED_HISTORY = 3
DIVISION_DIST_THRESHOLD = 120

AREA_DIVISION_RATIO = 1.8

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if ZIP_PATH and os.path.exists(ZIP_PATH) and (not os.path.exists(IMAGE_FOLDER) or len(os.listdir(IMAGE_FOLDER)) == 0):
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(IMAGE_FOLDER)

def find_image_files(folder):
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = []
    for root, dirs, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return sorted(files)

image_paths = find_image_files(IMAGE_FOLDER)
if len(image_paths) == 0:
    image_paths = find_image_files(".")
    if len(image_paths) == 0:
        raise SystemExit("No images found.")

print("Found frames:", len(image_paths))

# --- HELPER FUNCTIONS ---

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            img = np.array(Image.open(path).convert('L'))
        except Exception:
            return None
    return img

def robust_binarize(img):
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = exposure.rescale_intensity(img, out_range='uint8').astype(np.uint8)
    p = np.percentile(img, 99.0)
    bw = img < p
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=10)
    bw = morphology.remove_small_holes(bw, area_threshold=20)
    bw = morphology.binary_opening(bw, footprint=morphology.disk(2))
    return bw.astype(np.uint8)

def medoid_of_mask(mask, bbox):
    if mask is None or not np.any(mask):
        return None
    minr, minc, maxr, maxc = bbox
    sub = mask[minr:maxr, minc:maxc]
    if not np.any(sub):
        return None
    props = measure.regionprops(sub.astype(np.uint8))
    if len(props) > 0:
        cy, cx = props[0].centroid
        return (int(round(cy)) + minr, int(round(cx)) + minc)
    return None

def build_objects_dict(lab):
    objs = {}
    for pr in measure.regionprops(lab):
        if pr.area < MIN_AREA:
            continue
        med = medoid_of_mask(lab == pr.label, pr.bbox)
        if med is None:
            cy, cx = pr.centroid
            med = (int(round(cy)), int(round(cx)))
        objs[pr.label] = {
            'label': pr.label,
            'area': pr.area,
            'centroid': pr.centroid,
            'bbox': pr.bbox,
            'medoid': med,
            'mask': (lab == pr.label).astype(np.uint8)
        }
    return objs

def compute_iou(maskA, maskB):
    if maskA is None or maskB is None:
        return 0.0
    inter = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def in_lineage(name, root):
    if name is None:
        return False
    name = str(name)
    return name == root or name.startswith(root + "_")

def predict_from_history(history, current_frame, k=PRED_HISTORY):
    if not history:
        return None
    past = [h for h in history if h['frame'] < current_frame]
    if not past:
        return None
    past = past[-k:]
    if len(past) == 1:
        last = past[-1]
        return float(last['y']), float(last['x'])
    first = past[0]
    last = past[-1]
    dt = max(last['frame'] - first['frame'], 1)
    vy = (last['y'] - first['y']) / dt
    vx = (last['x'] - first['x']) / dt
    py = last['y'] + vy
    px = last['x'] + vx
    return float(py), float(px)

def snap_seed_to_peak(dt, mask_local, lr, lc, radius):
    h, w = mask_local.shape
    r_start = max(0, lr - radius)
    r_end = min(h, lr + radius + 1)
    c_start = max(0, lc - radius)
    c_end = min(w, lc + radius + 1)
    window = dt[r_start:r_end, c_start:c_end]
    if window.size == 0:
        return None, None
    if np.max(window) <= 0:
        return None, None
    max_idx = np.argmax(window)
    wr, wc = np.unravel_index(max_idx, window.shape)
    return r_start + wr, c_start + wc

def split_blob_with_seeds(
    lab, cid, seeds_global, prev_stats,
    erosion_iter, snap_radius, current_max_lab,
    frame_num=None, debug=False
):
    reason = [] 
    
    mask_c = (lab == cid)
    if not np.any(mask_c):
        reason.append("MASK_EMPTY -> blob has no pixels")
        print(f"[SPLIT FAIL] Frame {frame_num} Blob {cid}: {reason}")
        return current_max_lab, []

    coords = np.argwhere(mask_c)
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0) + 1
    mask_local = mask_c[minr:maxr, minc:maxc]

    total_area = mask_local.sum()
    if total_area < 10:
        reason.append("BLOB_TOO_SMALL -> area < 10px cannot meaningfully separate")

    prev_areas = np.array([prev_stats[n]['area'] for n in seeds_global.keys()], float)
    if prev_areas.sum() == 0:
        reason.append("PREV_STATS_ZERO -> no valid area history")

    area_ratios = prev_areas / (prev_areas.sum() + 1e-9)
    target_areas = (area_ratios * total_area).astype(int)

    mask_dt = mask_local.copy()
    if erosion_iter > 0:
        eroded = morphology.binary_erosion(mask_local, morphology.disk(erosion_iter))
        if eroded.sum() < 5:
            reason.append("EROSION_TOO_STRONG -> erosion collapsed structure")
        else:
            mask_dt = eroded

    dt = ndi.distance_transform_edt(mask_dt)

    local_seeds = {}
    snapping_fail = 0
    index_fail = 0
    too_close_seed = False

    sid = 1
    for nm,(gy,gx) in seeds_global.items():
        lr = int(round(gy)) - minr
        lc = int(round(gx)) - minc
        if lr<0 or lc<0 or lr>=mask_local.shape[0] or lc>=mask_local.shape[1]:
            index_fail += 1
            continue

        sr, sc = snap_seed_to_peak(dt, mask_dt, lr, lc, snap_radius)
        if sr is None:
            snapping_fail += 1
            continue

        local_seeds[nm] = (sid, sr, sc)
        sid += 1

    if len(local_seeds) < 2:
        if snapping_fail > 0: reason.append(f"SEED_SNAP_FAIL({snapping_fail})")
        if index_fail > 0: reason.append(f"SEED_OUT_OF_BOUNDS({index_fail})")
        if len(local_seeds) == 1: reason.append("ONLY_ONE_SEED_LEFT")
        if len(local_seeds) == 0: reason.append("NO_VALID_SEEDS")
        print(f"[SPLIT FAIL] Frame {frame_num} Blob {cid}: {reason}")
        return current_max_lab, []

    pts = [(sr,sc) for (_,sr,sc) in local_seeds.values()]
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            if np.hypot(pts[i][0]-pts[j][0],pts[i][1]-pts[j][1]) < 4:
                too_close_seed = True
    if too_close_seed:
        reason.append("SEEDS_TOO_CLOSE (<4px) -> watershed merges them")

    assign = np.zeros_like(mask_local,int)
    used = np.zeros_like(mask_local,bool)
    names_order=list(seeds_global.keys())

    for nm,(sid, sr, sc) in local_seeds.items():
        grow=np.zeros_like(mask_local,bool); grow[sr,sc]=True
        target_i = names_order.index(nm)
        target_pix = target_areas[target_i] if target_i<len(target_areas) else int(total_area/len(local_seeds))
        while grow.sum()<target_pix:
            front=ndi.binary_dilation(grow)&mask_local&(~grow)
            if not front.any():
                reason.append(f"GROW_STALLED({nm})")
                break
            grow[front]=True
        assign[grow]=sid; used|=grow

    rem=(~used)&mask_local
    if rem.any(): reason.append("REASSIGN_REMAINING -> initial Voronoi weak")

    seed_mark=assign.copy()
    split_refined=segmentation.watershed(-dt,seed_mark,mask=mask_local)
    pieces=np.unique(split_refined)[1:] 

    if len(pieces)<2:
        reason.append(f"WATERSHED_SINGLE_OUTPUT(n={len(pieces)}) -> unimodal DT or seeds collapsed")
        print(f"[SPLIT FAIL] Frame {frame_num} Blob {cid}: {reason}")
        return current_max_lab, []

    sub = lab[minr:maxr, minc:maxc]; sub[sub==cid]=0
    new_labels=[]
    for sid in pieces:
        current_max_lab+=1
        sub[split_refined==sid]=current_max_lab
        new_labels.append(current_max_lab)

    lab[minr:maxr, minc:maxc] = sub
    print(f"[SPLIT SUCCESS] Frame {frame_num} Blob {cid} -> pieces:{len(new_labels)} seeds:{len(local_seeds)}")
    return current_max_lab,new_labels

def make_unique_name(base, existing):
    if base not in existing:
        return base
    k = 1
    while True:
        candidate = f"{base}_{k}"
        if candidate not in existing:
            return candidate
        k += 1

def geometric_two_seeds_for_blob(lab, cid):
    coords = np.argwhere(lab == cid)
    if coords.shape[0] < 2:
        return None, None
    ys = coords[:, 0]
    xs = coords[:, 1]

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    yr = y_max - y_min
    xr = x_max - x_min

    if xr >= yr:
        mid = 0.5 * (x_min + x_max)
        left = coords[xs <= mid]
        right = coords[xs > mid]
        if len(left) == 0 or len(right) == 0:
            return None, None
        y1, x1 = np.mean(left, axis=0)
        y2, x2 = np.mean(right, axis=0)
    else:
        mid = 0.5 * (y_min + y_max)
        top = coords[ys <= mid]
        bottom = coords[ys > mid]
        if len(top) == 0 or len(bottom) == 0:
            return None, None
        y1, x1 = np.mean(top, axis=0)
        y2, x2 = np.mean(bottom, axis=0)

    return (float(y1), float(x1)), (float(y2), float(x2))
frames_labeled = []
frames_objects = []
processed_paths = []
name_map = []
next_global_id = 1
track_history = {}

print("Starting Motion-Aware Split Tracking with Division-Overlap Detection...")

for t, p in enumerate(image_paths):
    img = read_gray(p)
    if img is None:
        continue
    bw = robust_binarize(img)

    lab = measure.label(bw, connectivity=1)

    props = measure.regionprops(lab)
    valid_mask = np.zeros_like(lab, dtype=bool)
    for pr in props:
        if pr.area >= MIN_AREA:
            valid_mask[lab == pr.label] = True
    lab = measure.label(valid_mask, connectivity=1)

    cur_objs = build_objects_dict(lab)

    print(f"\n=== DEBUG FRAME {t} ===")
    print("Objects:", list(cur_objs.keys()))
    predicted_hits = {}
    forced_assignments = {}  
    if t >= TRAIN_FRAMES and t > 0 and len(cur_objs) > 0 and len(name_map) > 0:
        prev_names = name_map[t-1]
        active_names = set(prev_names.values())

        predictions = {}
        h, w = lab.shape
        cur_med_pts = np.array([cur_objs[cid]['medoid'] for cid in cur_objs.keys()]) if cur_objs else np.zeros((0, 2))

        for nm in active_names:
            hist = track_history.get(nm, [])
            pred = predict_from_history(hist, current_frame=t)
            if pred is None:
                continue
            py, px = pred
            gy, gx = int(round(py)), int(round(px))

            if gy < 0 or gx < 0 or gy >= h or gx >= w:
                continue
            if len(cur_med_pts) > 0:
                dists = np.hypot(cur_med_pts[:, 0] - py, cur_med_pts[:, 1] - px)
                min_d = float(dists.min())
                if min_d > DIVISION_DIST_THRESHOLD:
                    continue

            cid = int(lab[gy, gx])

            if cid == 0:
                search_r = 30
                y_min, y_max = max(0, gy - search_r), min(h, gy + search_r + 1)
                x_min, x_max = max(0, gx - search_r), min(w, gx + search_r + 1)
                sub_lab = lab[y_min:y_max, x_min:x_max]
                if np.any(sub_lab > 0):
                    ys, xs = np.where(sub_lab > 0)
                    ys_g = ys + y_min
                    xs_g = xs + x_min
                    d2 = (ys_g - gy)**2 + (xs_g - gx)**2
                    nearest_idx = np.argmin(d2)
                    if d2[nearest_idx] < search_r**2:
                        cid = int(lab[ys_g[nearest_idx], xs_g[nearest_idx]])

            if cid <= 0:
                continue

            predictions[nm] = (gy, gx, cid)
            predicted_hits.setdefault(cid, []).append(nm)

        print("Predicted_hits:", predicted_hits)
        merge_cids = [cid for cid, nms in predicted_hits.items() if len(nms) >= 2]
        print("MERGE_CIDS:", merge_cids)

        if merge_cids:
            print(f"Frame {t}: Pre-naming split of {len(merge_cids)} merged blobs.")
            current_max_lab = lab.max()
            
            for cid in merge_cids:
                parent_names = predicted_hits[cid]
                prev_stats = {}
                seeds_global = {}

                for nm in parent_names:
                    hist = track_history.get(nm, [])
                    if not hist:
                        continue
                    last = hist[-1]
                    prev_stats[nm] = {
                        'area': last['area'],
                        'y': last['y'],
                        'x': last['x']
                    }
                    gy, gx, _ = predictions[nm]
                    seeds_global[nm] = (gy, gx)

                if len(seeds_global) <= 1:
                    continue
                current_max_lab, new_labels = split_blob_with_seeds(
                    lab, cid, seeds_global, prev_stats,
                    EROSION_ITERATIONS, SEED_SNAP_RADIUS, current_max_lab, frame_num=t
                )
                temp_objs = build_objects_dict(lab)
                used_new_labels = set()
                
                for nm in parent_names:
                    if nm not in seeds_global: continue
                    sy, sx = seeds_global[nm]
                    
                    best_match_cid = None
                    min_dist_to_seed = float('inf')

                    for new_cid in new_labels:
                        if new_cid in used_new_labels: continue
                        if new_cid not in temp_objs: continue
                        med = temp_objs[new_cid]['medoid']
                        d = np.hypot(med[0] - sy, med[1] - sx)
                        if d < MAX_MATCH_DIST and d < min_dist_to_seed:
                            min_dist_to_seed = d
                            best_match_cid = new_cid
                    
                    if best_match_cid is not None:
                        forced_assignments[best_match_cid] = nm
                        used_new_labels.add(best_match_cid)
                        print(f"   -> Forced Match: New Label {best_match_cid} is Track '{nm}'")
            
            cur_objs = build_objects_dict(lab)
    frames_labeled.append(lab)
    frames_objects.append(cur_objs)
    processed_paths.append(p)
    name_map.append({})

    cur_ids = list(cur_objs.keys())
    assigned_cur_indices = set()
    if forced_assignments:
        for cid, forced_name in forced_assignments.items():
            if cid in cur_objs:
                name_map[t][cid] = forced_name
                if cid in cur_ids:
                    assigned_cur_indices.add(cur_ids.index(cid))
        print(f"Frame {t}: Applied forced assignments: {forced_assignments}")

    if t == 0:
        for cid in cur_ids:
            nm = str(cid)
            name_map[t][cid] = nm
            if cid >= next_global_id:
                next_global_id = cid + 1
    else:
        prev_objs = frames_objects[t-1]
        prev_ids = list(prev_objs.keys())

        if len(prev_ids) > 0 and len(cur_ids) > 0:
            iou_mat = np.zeros((len(prev_ids), len(cur_ids)), dtype=float)
            for i, pid in enumerate(prev_ids):
                for j, cid in enumerate(cur_ids):
                    iou_mat[i, j] = compute_iou(prev_objs[pid]['mask'], cur_objs[cid]['mask'])

            cur_med_pts = np.array([cur_objs[cid]['medoid'] for cid in cur_ids])

            for i, pid in enumerate(prev_ids):
                matches = []
                for j, cid in enumerate(cur_ids):
                    if j in assigned_cur_indices:
                        continue
                    if iou_mat[i, j] > 0.1:
                        matches.append(cid)

                matches.sort(key=lambda x: cur_objs[x]['area'], reverse=True)
                p_name = name_map[t-1].get(pid, str(pid))

                force_division = False
                hist = track_history.get(p_name, [])
                pred = predict_from_history(hist, current_frame=t)
                if pred is not None and len(cur_med_pts) > 0:
                    py, px = pred
                    dists_pred = np.hypot(cur_med_pts[:, 0] - py, cur_med_pts[:, 1] - px)
                    min_d_pred = float(dists_pred.min())
                    if min_d_pred > DIVISION_DIST_THRESHOLD:
                        force_division = True

                if len(matches) == 1 and not force_division:
                    cid = matches[0]
                    name_map[t][cid] = p_name
                    assigned_cur_indices.add(cur_ids.index(cid))

                elif len(matches) >= 1 and (force_division or len(matches) > 1):
                    for k, cid in enumerate(matches, 1):
                        new_name = f"{p_name}_{k}"
                        while new_name in name_map[t].values():
                            new_name = new_name + "_"
                        name_map[t][cid] = new_name
                        assigned_cur_indices.add(cur_ids.index(cid))
        unassigned_indices = [j for j in range(len(cur_ids)) if j not in assigned_cur_indices]
        if unassigned_indices and len(prev_ids) > 0:
            cur_pts = [cur_objs[cur_ids[j]]['medoid'] for j in unassigned_indices]
            prev_pts = [prev_objs[pid]['medoid'] for pid in prev_ids]

            if len(cur_pts) > 0 and len(prev_pts) > 0:
                dists = cdist(cur_pts, prev_pts)
                for row, j in enumerate(unassigned_indices):
                    cid = cur_ids[j]
                    closest_prev_idx = int(np.argmin(dists[row]))
                    min_d = dists[row, closest_prev_idx]
                    
                    if min_d < MAX_MATCH_DIST:
                        pid = prev_ids[closest_prev_idx]
                        p_name = name_map[t-1].get(pid, str(pid))
                        final_name = p_name
                        if final_name in name_map[t].values():
                            k = 1
                            while f"{p_name}_r{k}" in name_map[t].values():
                                k += 1
                            final_name = f"{p_name}_r{k}"
                        
                        name_map[t][cid] = final_name
                        print(f"Frame {t}: Rescue {cid} -> {final_name}")
                    else:
                        name_map[t][cid] = str(next_global_id)
                        next_global_id += 1

        for cid in cur_ids:
            if cid not in name_map[t]:
                name_map[t][cid] = str(next_global_id)
                next_global_id += 1

    print(f"Frame {t}: Initial assignments: {dict(name_map[t])}")
    if t >= TRAIN_FRAMES and t > 0:
        lab = frames_labeled[t]
        cur_objs = frames_objects[t]
        active_names_now = set(name_map[t].values())
        h, w = lab.shape

        preserved_assignments = name_map[t].copy()

        post_predictions = {}
        post_hits = {}
        for nm in active_names_now:
            hist = track_history.get(nm, [])
            pred = predict_from_history(hist, current_frame=t)
            if pred is None:
                continue
            py, px = pred
            gy, gx = int(round(py)), int(round(px))
            if gy < 0 or gx < 0 or gy >= h or gx >= w:
                continue
            cid = int(lab[gy, gx])
            if cid == 0:
                search_r = 30
                y_min, y_max = max(0, gy - search_r), min(h, gy + search_r + 1)
                x_min, x_max = max(0, gx - search_r), min(w, gx + search_r + 1)
                sub_lab = lab[y_min:y_max, x_min:x_max]
                if np.any(sub_lab > 0):
                    ys, xs = np.where(sub_lab > 0)
                    ys_g = ys + y_min
                    xs_g = xs + x_min
                    d2 = (ys_g - gy)**2 + (xs_g - gx)**2
                    nearest_idx = np.argmin(d2)
                    if d2[nearest_idx] < search_r**2:
                        cid = int(lab[ys_g[nearest_idx], xs_g[nearest_idx]])
            if cid <= 0:
                continue
            post_predictions[nm] = (gy, gx, cid)
            post_hits.setdefault(cid, []).append(nm)

        print("Post_hits:", post_hits)

        split_seed_dict = {}
        split_prev_stats = {}
        existing_names_global = set(track_history.keys()) | set(active_names_now)
        collision_cids = [cid for cid, names in post_hits.items() if len(names) >= 2]
        for cid in collision_cids:
            names = post_hits[cid]
            seeds_global = {}
            prev_stats = {}
            for nm in names:
                hist = track_history.get(nm, [])
                if not hist:
                    continue
                last = hist[-1]
                prev_stats[nm] = {'area': last['area'], 'y': last['y'], 'x': last['x']}
                gy, gx, _ = post_predictions[nm]
                seeds_global[nm] = (gy, gx)
            if len(seeds_global) > 1:
                split_seed_dict[cid] = seeds_global
                split_prev_stats[cid] = prev_stats
                print(f" Frame {t}: POST-NAMING MERGE SPLIT -- tracks={names}, blob={cid}")
        division_cands = []
        for nm in active_names_now:
            if '_' not in nm or not nm.startswith(USER_REQUESTED_NAME):
                continue
            parent = nm.split('_')[0]
            if parent not in track_history or len(track_history[parent]) == 0:
                continue
            if nm not in post_predictions:
                continue
            gy, gx, cid = post_predictions[nm]
            assigned_name = name_map[t].get(cid, "")
            if cid > 0 and nm != assigned_name:
                division_cands.append((cid, nm, parent))

        print("Division candidates:", division_cands)

        for cid, nm, parent in division_cands:
            print(f"Frame {t}: DIVISION_OVERLAP_CHECK -- daughter={nm}, parent={parent}, blob={cid}")
            seeds_geo = geometric_two_seeds_for_blob(lab, cid)
            if seeds_geo == (None, None):
                continue
            (y1, x1), (y2, x2) = seeds_geo
            child1 = make_unique_name(f"{parent}_1", existing_names_global)
            child2 = make_unique_name(f"{parent}_2", existing_names_global)
            existing_names_global.add(child1)
            existing_names_global.add(child2)

            seeds_global = {child1: (y1, x1), child2: (y2, x2)}
            last_parent = track_history[parent][-1]
            prev_stats = {
                child1: {'area': last_parent['area']/2.0, 'y': last_parent['y'], 'x': last_parent['x']},
                child2: {'area': last_parent['area']/2.0, 'y': last_parent['y'], 'x': last_parent['x']},
            }
            split_seed_dict[cid] = seeds_global
            split_prev_stats[cid] = prev_stats
            print(f"Frame {t}: DIVISION_OVERLAP -- parent={parent}, daughter={nm}, blob={cid}")
        for nm, (gy, gx, cid) in post_predictions.items():
            if cid in split_seed_dict:
                continue
            if cid not in cur_objs:
                continue
            hist = track_history.get(nm, [])
            if not hist:
                continue
            prev_area = hist[-1]['area']
            cur_area = cur_objs[cid]['area']
            if prev_area <= 0:
                continue
            ratio = cur_area / prev_area
            if ratio >= AREA_DIVISION_RATIO * 0.7:
                seeds_geo = geometric_two_seeds_for_blob(lab, cid)
                if seeds_geo == (None, None):
                    continue
                (y1, x1), (y2, x2) = seeds_geo
                base1 = f"{nm}_1"
                base2 = f"{nm}_2"
                child1 = make_unique_name(base1, existing_names_global)
                child2 = make_unique_name(base2, existing_names_global)
                existing_names_global.add(child1)
                existing_names_global.add(child2)

                seeds_global = {child1: (y1, x1), child2: (y2, x2)}
                prev_stats = {
                    child1: {'area': prev_area / 2.0, 'y': hist[-1]['y'], 'x': hist[-1]['x']},
                    child2: {'area': prev_area / 2.0, 'y': hist[-1]['y'], 'x': hist[-1]['x']},
                }
                split_seed_dict[cid] = seeds_global
                split_prev_stats[cid] = prev_stats
                print(f" Frame {t}: AREA ANOMALY DIVISION -- parent={nm}, blob={cid}, ratio={ratio:.2f}")

        if split_seed_dict:
            print(f"Frame {t}: Executing {len(split_seed_dict)} splits...")
            current_max_lab = lab.max()
            cid_new_labels_map = {}
            for cid in split_seed_dict.keys():
                if cid in name_map[t]:
                    print(f"Frame {t}: Clearing split target {cid} (was {name_map[t][cid]})")
                    del name_map[t][cid]

            for cid, seeds_global in split_seed_dict.items():
                prev_stats = split_prev_stats[cid]
                current_max_lab, new_labels = split_blob_with_seeds(
                    lab, cid, seeds_global, prev_stats,
                    EROSION_ITERATIONS, SEED_SNAP_RADIUS, current_max_lab, frame_num=t
                )
                cid_new_labels_map[cid] = (new_labels, seeds_global)

            cur_objs = build_objects_dict(lab)
            frames_objects[t] = cur_objs
            frames_labeled[t] = lab

            for cid, (new_labels, seeds_global) in cid_new_labels_map.items():
                if not new_labels:
                    continue
                for new_cid in new_labels:
                    if new_cid not in cur_objs:
                        continue
                    med = cur_objs[new_cid]['medoid']
                    my, mx = med[0], med[1]
                    best_name = None
                    best_dist = float('inf')
                    for nm, (sy, sx) in seeds_global.items():
                        d = np.hypot(my - sy, mx - sx)
                        if d < best_dist:
                            best_dist = d
                            best_name = nm
                    if best_name is not None:
                        name_map[t][new_cid] = best_name
                        print(f"   -> Frame {t}: new piece {new_cid} -> {best_name}")

            for cid, nm in preserved_assignments.items():
                if cid not in name_map[t] and cid in cur_objs:
                    name_map[t][cid] = nm
                    print(f"Frame {t}: RESTORED {cid} -> {nm}")

            print(f"Frame {t}: Final assignments: {dict(name_map[t])}")

    for cid, obj in frames_objects[t].items():
        nm = name_map[t].get(cid, None)
        if nm is None:
            continue
        med = obj['medoid']
        track_history.setdefault(nm, []).append({
            'frame': t,
            'y': float(med[0]),
            'x': float(med[1]),
            'area': float(obj['area']),
            'label': cid
        })


frames_gif = []
for t, p in enumerate(processed_paths):
    base = Image.open(p).convert('RGB')
    draw = ImageDraw.Draw(base)

    for lbl, obj in frames_objects[t].items():
        nm = str(name_map[t].get(lbl, "?"))
        if nm.startswith("1"):
            color = (0, 255, 0)
            
            med = obj['medoid']
            r, c = int(med[0]), int(med[1])
            draw.ellipse((c-2, r-2, c+2, r+2), fill=color)
            

    frames_gif.append(np.array(base))
gif_out = os.path.join(OUTPUT_FOLDER, f"tracked_post_split_only_1.gif")

if frames_gif:
    imageio.mimsave(gif_out, frames_gif, duration=0.15, loop=0)
    print("Saved ->", gif_out)
print("Done.")


print("Press 'Q' when done → GIF highlighting lineage will be generated")

selected_parent = None  
win = "Frame Viewer"
cv2.namedWindow(win)

current_frame = 0
total_frames = len(processed_paths)

frame_medoid_map = {}
for t in range(total_frames):
    frame_medoid_map[t] = []
    for cid, obj in frames_objects[t].items():
        nm = name_map[t][cid]
        frame_medoid_map[t].append((cid, nm, obj['medoid']))

def set_frame(v):
    global current_frame
    current_frame = v

cv2.createTrackbar("Frame", win, 0, total_frames-1, set_frame)

def click(event, x, y, flags, param):
    global selected_parent
    if event == cv2.EVENT_LBUTTONDOWN:
        medoids = frame_medoid_map[current_frame]
        for cid, nm, (r,c) in medoids:
            if abs(r-y) < 5 and abs(c-x) < 5:  
                selected_parent = nm.split("_")[0]  
                print(f"Parent chosen → {selected_parent}")
                return

cv2.setMouseCallback(win, click)

def processing_cycle(img):
    arr = np.array(img, dtype=np.float32)
    for _ in range(45):                     
        arr = ndi.gaussian_filter(arr, sigma=0.8)
        arr = np.abs(np.fft.ifft2(np.fft.fft2(arr)))   
    return float(arr.mean())



while True:
    frame = cv2.imread(processed_paths[current_frame])
    disp = frame.copy()
    for cid, nm, (r,c) in frame_medoid_map[current_frame]:
        cv2.circle(disp,(c,r),3,(0,255,0),-1)
        if selected_parent and nm.startswith(selected_parent):
            cv2.circle(disp,(c,r),6,(0,0,255),1)

    cv2.imshow(win, disp)
    key = cv2.waitKey(30)

    if key in [ord('q'), ord('Q')]:
        break

cv2.destroyAllWindows()


if not selected_parent:
    print("No parent selected. No GIF created.")
    exit()

print(f"GENERATING LINEAGE GIF FOR ROOT: {selected_parent}")

gif_frames = []
out = os.path.join(OUTPUT_FOLDER, f"LINEAGE_{selected_parent}.gif")

for t, path in enumerate(processed_paths):

    processing_cycle(Image.open(path))   
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for cid,obj in frames_objects[t].items():
        nm = name_map[t][cid]
        if nm.startswith(selected_parent):
            r,c=obj['medoid']
            draw.ellipse((c-3,r-3,c+3,r+3),fill=(255,0,0))

    gif_frames.append(np.array(img))
    print(f"Frame {t} processed")

imageio.mimsave(out,gif_frames,duration=0.2,loop=0)
print("GIF SAVED ->",out)
