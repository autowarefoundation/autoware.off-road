#!/usr/bin/env python3
import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import time
import copy
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor

# -------------------- Parse Arguments --------------------
parser = argparse.ArgumentParser(
    description="Interactive SAM-based labeling tool (paint/erase, undo, overlay toggle, autosave, autoload, click-to-update, auto-label)"
)
parser.add_argument("-p", "--sam_checkpoint", required=True)
parser.add_argument("-i", "--input_folder", required=True)
parser.add_argument("-o", "--output_folder", required=True)
parser.add_argument("-c", "--color_map", required=True)
parser.add_argument("--no-fill", action="store_true")
args = parser.parse_args()

# -------------------- Safety Check --------------------
if os.path.abspath(args.input_folder) == os.path.abspath(args.output_folder):
    raise ValueError("Input and output folder paths must be different to avoid overwriting images.")

os.makedirs(args.output_folder, exist_ok=True)

# -------------------- Config --------------------
fill_undefined = not args.no_fill
overlay_visible = True
CLICK_HOLD_THRESHOLD = 0.25
BRUSH_MIN, BRUSH_MAX = 5, 50    # min/max for both brush and eraser
brush_radius = 20               # initial brush radius
eraser_radius = 20              # initial eraser radius

# Fill small undefined holes (in unlabeled regions) narrower than this many pixels
FILL_HOLE_WIDTH = 30  # Morphological closing kernel width (in pixels)

# Preview of radius when using +/- or wheel
radius_preview_tool = None      # 'brush' or 'eraser'
radius_preview_expire = 0.0

# -------------------- Load Color Config --------------------
with open(args.color_map, "r") as f:
    config = yaml.safe_load(f)

# reverse RGB->BGR for OpenCV
id_to_color = {int(k): tuple(v[::-1]) for k, v in config["id_to_color_mapping"].items()}
color_to_id = {tuple(v): k for k, v in id_to_color.items()}

if 0 not in id_to_color:
    raise ValueError("Color map must include id=0 (for fill_undefined).")

# -------------------- Initialize SAM --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# -------------------- Load Images --------------------
image_files = sorted([
    os.path.join(args.input_folder, f)
    for f in os.listdir(args.input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
if not image_files:
    raise ValueError("No images found!")

# -------------------- Globals --------------------
current_index = 0
image = cv2.imread(image_files[current_index])
h, w = image.shape[:2]

mask_overlay = np.zeros_like(image, np.uint8)
mask_per_id: dict[int, np.ndarray] = {}
points_per_id: dict[int, list[tuple[int, int]]] = defaultdict(list)
history = []

drawing = False            # manual brush state (LMB hold)
manual_mode = False
manual_pushed = False
erasing = False            # continuous eraser state (MMB hold)
erase_pushed = False
filling = False            # continuous fill (Ctrl + LMB hold)
erase_filling = False      # continuous erase-fill (Ctrl + MMB hold)

mouse_down_time = 0.0
last_mouse_pos = (0, 0)
modified = False
selected_id = 0
selected_color = id_to_color[selected_id]

# -------------------- Controls Info --------------------
print("============================================================")
print("Interactive SAM Labeling Tool — Controls & Shortcuts")
print("============================================================")
print("Mouse:")
print("  • Short Left Click (<0.25s): Add SAM prompt point for CURRENT color ID")
print("  • Hold Left Click (≥0.25s): Manual paint (yellow brush outline)")
print("  • Middle Button Drag: Erase mask (red eraser outline)")
print("  • Ctrl + Left Button (hold & move): FILL mode — flood-fill connected region of SAME LABEL")
print("  • Ctrl + Middle Button (hold & move): ERASE-FILL mode — erase connected region of SAME LABEL")
print()
print("Keyboard:")
print("  • 0–9: Switch color ID manually")
print("  • + / -: Increase / decrease radius of ACTIVE tool by 5 px (min 5, max 50)")
print("           (Active tool = eraser if erasing / erase-fill active, otherwise brush)")
print("  • H: Toggle mask overlay visibility")
print("  • Z: Undo last operation")
print("  • N / B: Next / Previous image (auto-saves if modified)")
print("  • M: Move to NEXT image and")
print("       auto-label it using the current cursor position as a SAM prompt")
print("  • C: Clear ALL labels and SAM points for the current image")
print("  • /: Jump to NEXT image that has NO segmentation mask")
print("  • ESC: Save current mask and exit")
print()
print("Notes:")
print(f"  - FILL_HOLE_WIDTH = {FILL_HOLE_WIDTH} px (morphological closing on undefined regions)")
print("  - Preloaded masks can be refined directly (no auto color switching).")
print("============================================================")

# -------------------- Helper Functions --------------------
def clamp(v):
    return max(BRUSH_MIN, min(BRUSH_MAX, int(v)))

def basename(idx=None):
    if idx is None:
        idx = current_index
    return os.path.splitext(os.path.basename(image_files[idx]))[0]

def mask_path(idx=None):
    if idx is None:
        idx = current_index
    return os.path.join(args.output_folder, basename(idx) + ".png")

def print_image_id(prefix=""):
    # 1-based index, total number of images
    if prefix:
        print(prefix, end="")
    print(f"{current_index + 1} / {len(image_files)}")

def push_history():
    history.append((copy.deepcopy(mask_per_id), mask_overlay.copy()))

def undo():
    global mask_per_id, mask_overlay, modified
    if history:
        prev_masks, prev_overlay = history.pop()
        mask_per_id = prev_masks
        mask_overlay[:] = prev_overlay
        modified = True
        print("Undo.")
    else:
        print("Nothing to undo.")

def rebuild_overlay():
    mask_overlay[:] = 0
    for cid, m in mask_per_id.items():
        mask_overlay[m] = id_to_color[cid]

def save_mask():
    global modified
    if not modified:
        return
    out = mask_overlay.copy()
    if fill_undefined:
        undef = (out == 0).all(-1)
        out[undef] = id_to_color[0]
    cv2.imwrite(mask_path(), out)
    print(f"Saved {mask_path()}")
    modified = False

def load_mask():
    global mask_per_id, mask_overlay, modified
    p = mask_path()
    if not os.path.exists(p):
        mask_per_id.clear()
        mask_overlay[:] = 0
        print(f"No existing mask for {basename()}")
        return
    m = cv2.imread(p)
    if m is None:
        print(f"Failed to load {p}")
        return
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_overlay[:] = m
    mask_per_id.clear()
    for cid, col in id_to_color.items():
        match = np.all(m == np.array(col, np.uint8), axis=-1)
        if np.any(match):
            mask_per_id[cid] = match
    modified = False
    print(f"Loaded existing mask: {p}")

def reset_image():
    global mask_overlay, mask_per_id, points_per_id, history, modified, h, w
    h, w = image.shape[:2]
    mask_overlay[:] = 0
    mask_per_id.clear()
    points_per_id.clear()
    history.clear()
    modified = False
    load_mask()

def fill_small_undefined_holes(mask_dict: dict[int, np.ndarray], kernel_width: int):
    """
    Fill small unlabeled gaps (holes) inside the union of all class masks
    using morphological closing. New pixels are assigned to `selected_id`.
    """
    if kernel_width <= 0 or not mask_dict:
        return

    # Union of all labeled pixels
    combined = np.zeros_like(next(iter(mask_dict.values())), dtype=np.uint8)
    for m in mask_dict.values():
        combined |= m.astype(np.uint8)  # 1 where labeled, 0 where undefined

    # Run closing on the union
    combined_255 = combined * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_width))
    closed = cv2.morphologyEx(combined_255, cv2.MORPH_CLOSE, kernel)

    # New "on" pixels created by closing (were 0 before, >0 now)
    filled_areas = (closed > 0) & (combined == 0)

    if np.any(filled_areas):
        # Make sure we have a mask for selected_id
        if selected_id not in mask_dict:
            mask_dict[selected_id] = np.zeros_like(combined, dtype=bool)
        mask_dict[selected_id][filled_areas] = True

def run_sam(points_list):
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pts = np.array(points_list)
    lbl = np.ones(len(points_list))
    masks, _, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=False)
    mask = masks[0].astype(bool)

    if selected_id not in mask_per_id:
        mask_per_id[selected_id] = mask
    else:
        mask_per_id[selected_id] = np.logical_or(mask_per_id[selected_id], mask)

    # Fill undefined holes using morphological closing
    fill_small_undefined_holes(mask_per_id, FILL_HOLE_WIDTH)
    rebuild_overlay()

def goto_next_image():
    """Go to next image (cyclic), saving current mask first."""
    global current_index, image
    save_mask()
    current_index = (current_index + 1) % len(image_files)
    image = cv2.imread(image_files[current_index])
    reset_image()
    print(f"Next: {image_files[current_index]}")
    print_image_id()

def goto_prev_image():
    global current_index, image
    save_mask()
    current_index = (current_index - 1) % len(image_files)
    image = cv2.imread(image_files[current_index])
    reset_image()
    print(f"Prev: {image_files[current_index]}")
    print_image_id()

def goto_next_unlabeled_image():
    """
    Jump to the next image (cyclic) that has NO saved mask file.
    Uses 1-based image indexing when printing.
    """
    global current_index, image
    total = len(image_files)
    start_index = current_index

    for step in range(1, total + 1):
        idx = (start_index + step) % total
        if not os.path.exists(mask_path(idx)):
            # Found an unlabeled image
            save_mask()
            current_index = idx
            image = cv2.imread(image_files[current_index])
            reset_image()
            print(f"Next unlabeled: {image_files[current_index]}")
            print_image_id()
            return

    # If we get here, all images have masks
    print("All images already have segmentation masks.")
    print_image_id()

def union_labeled_mask():
    """Union of all labeled pixels across classes."""
    if not mask_per_id:
        return np.zeros((h, w), dtype=np.uint8)
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in mask_per_id.values():
        combined |= m.astype(np.uint8)
    return combined

def get_label_map():
    """
    Build an integer label map from mask_per_id:
      - 0 means unlabeled
      - k means pixel belongs to class id k
    """
    label_map = np.zeros((h, w), dtype=np.int32)
    for cid, m in mask_per_id.items():
        label_map[m] = cid
    return label_map

def disk_mask(cx, cy, radius):
    """Return a boolean mask of a filled disk centered at (cx, cy)."""
    if radius <= 0:
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return dist2 <= radius * radius

def region_fill_from_brush(cx, cy, radius):
    """
    FILL mode (Ctrl + LMB drag):

    - Look at the label_id (color_id) of the pixel under the cursor.
      (0 = unlabeled, 1..N = class IDs)
    - Take all pixels inside the brush disk as seeds.
    - Consider only pixels whose label_id == label_id_at_cursor.
    - Flood-fill the connected region of that label that touches the seeds.
    - Reassign that whole region to `selected_id`.
    """
    global mask_per_id, mask_overlay

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return

    label_map = get_label_map()
    source_id = label_map[cy, cx]

    # All pixels that currently have the same label_id as under the cursor
    same_label = (label_map == source_id)
    if not np.any(same_label):
        return

    disk = disk_mask(cx, cy, radius)
    seeds = disk & same_label
    if not np.any(seeds):
        return

    num_labels, labels = cv2.connectedComponents(same_label.astype(np.uint8))
    seed_labels = np.unique(labels[seeds])
    if seed_labels.size == 0:
        return

    fill_region = np.isin(labels, seed_labels) & same_label
    if not np.any(fill_region):
        return

    # If we are already that ID, nothing to do
    if source_id == selected_id:
        return

    # Remove region from old source_id, if it was labeled
    if source_id in mask_per_id:
        src_mask = mask_per_id[source_id]
        src_mask[fill_region] = False
        if not np.any(src_mask):
            del mask_per_id[source_id]

    # Assign region to selected_id
    if selected_id not in mask_per_id:
        mask_per_id[selected_id] = np.zeros((h, w), dtype=bool)
    mask_per_id[selected_id][fill_region] = True

    rebuild_overlay()

def erase_fill_from_brush(cx, cy, radius):
    """
    ERASE-FILL mode (Ctrl + MMB drag):

    - Look at the label_id (color_id) of the pixel under the cursor.
    - Take all pixels inside the eraser disk as seeds.
    - Consider only pixels whose label_id == label_id_at_cursor.
    - Flood-fill the connected region of that label that touches the seeds.
    - Erase that entire region (set to unlabeled).
    """
    global mask_per_id, mask_overlay

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return

    label_map = get_label_map()
    source_id = label_map[cy, cx]

    # 0 → unlabeled, nothing to erase
    if source_id == 0:
        return

    same_label = (label_map == source_id)
    if not np.any(same_label):
        return

    disk = disk_mask(cx, cy, radius)
    seeds = disk & same_label
    if not np.any(seeds):
        return

    num_labels, labels = cv2.connectedComponents(same_label.astype(np.uint8))
    seed_labels = np.unique(labels[seeds])
    if seed_labels.size == 0:
        return

    erase_region = np.isin(labels, seed_labels) & same_label
    if not np.any(erase_region):
        return

    # Remove from that source_id mask
    if source_id in mask_per_id:
        src_mask = mask_per_id[source_id]
        src_mask[erase_region] = False
        if not np.any(src_mask):
            del mask_per_id[source_id]

    # Clear overlay in that region
    mask_overlay[erase_region] = 0

    rebuild_overlay()

# -------------------- Mouse Callback --------------------
def mouse_callback(event, x, y, flags, param):
    global drawing, manual_mode, manual_pushed, erasing, erase_pushed
    global filling, erase_filling
    global mouse_down_time, last_mouse_pos, modified
    global brush_radius, eraser_radius, radius_preview_tool, radius_preview_expire

    last_mouse_pos = (x, y)

    ctrl_down = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
    lbutton_down = bool(flags & cv2.EVENT_FLAG_LBUTTON)
    mbutton_down = bool(flags & cv2.EVENT_FLAG_MBUTTON)

    # Ctrl + Mouse Wheel: adjust active tool radius
    if event == cv2.EVENT_MOUSEWHEEL and ctrl_down:
        delta = 1 if flags > 0 else -1
        if erasing or erase_filling:
            eraser_radius = clamp(eraser_radius + delta * 2)
            print(f"Eraser radius: {eraser_radius}")
            radius_preview_tool = 'eraser'
        else:
            brush_radius = clamp(brush_radius + delta * 2)
            print(f"Brush radius: {brush_radius}")
            radius_preview_tool = 'brush'
        radius_preview_expire = time.time() + 0.7
        return

    # Ctrl + Left Button: continuous FILL MODE
    if event == cv2.EVENT_LBUTTONDOWN and ctrl_down:
        filling = True
        drawing = False
        manual_mode = False
        manual_pushed = False
        push_history()
        region_fill_from_brush(x, y, brush_radius)
        modified = True
        return

    # Ctrl + Middle Button: continuous ERASE-FILL MODE
    if event == cv2.EVENT_MBUTTONDOWN and ctrl_down:
        erase_filling = True
        erasing = False
        erase_pushed = False
        push_history()
        erase_fill_from_brush(x, y, eraser_radius)
        modified = True
        return

    if event == cv2.EVENT_LBUTTONDOWN and not ctrl_down:
        drawing = True
        manual_mode = False
        manual_pushed = False
        mouse_down_time = time.time()

    elif event == cv2.EVENT_MOUSEMOVE:
        # Continuous fill while Ctrl + LMB held
        if filling and ctrl_down and lbutton_down:
            region_fill_from_brush(x, y, brush_radius)
            modified = True

        # Continuous erase-fill while Ctrl + MMB held
        elif erase_filling and ctrl_down and mbutton_down:
            erase_fill_from_brush(x, y, eraser_radius)
            modified = True

        # Manual paint (no Ctrl)
        elif drawing and manual_mode:
            if not manual_pushed:
                push_history()
                manual_pushed = True
            tmp = np.zeros((h, w), np.uint8)
            cv2.circle(tmp, (x, y), brush_radius, 1, -1)
            area = tmp.astype(bool)
            mask_per_id.setdefault(selected_id, np.zeros((h, w), bool))[area] = True
            mask_overlay[area] = selected_color
            rebuild_overlay()
            modified = True

        # Simple eraser (no Ctrl)
        elif erasing:
            if not erase_pushed:
                push_history()
                erase_pushed = True
            tmp = np.zeros((h, w), np.uint8)
            cv2.circle(tmp, (x, y), eraser_radius, 1, -1)
            area = tmp.astype(bool)
            for cid in list(mask_per_id.keys()):
                mask_per_id[cid][area] = False
                if not np.any(mask_per_id[cid]):
                    del mask_per_id[cid]
            mask_overlay[area] = 0
            rebuild_overlay()
            modified = True

    elif event == cv2.EVENT_LBUTTONUP:
        # stop continuous fill if it was active
        filling = False

        # normal LMB release logic
        if drawing:
            drawing = False
            elapsed = time.time() - mouse_down_time
            if elapsed < CLICK_HOLD_THRESHOLD:
                # Short click: SAM point
                push_history()
                points_per_id[selected_id].append((x, y))
                print(f"Added SAM point ({x},{y}) for id={selected_id}")
                run_sam(points_per_id[selected_id])
                modified = True
            else:
                manual_mode = False
                manual_pushed = False

    elif event == cv2.EVENT_MBUTTONDOWN and not ctrl_down:
        erasing = True
        erase_pushed = False
        print("Eraser ON")

    elif event == cv2.EVENT_MBUTTONUP:
        # stop continuous erase-fill and simple eraser
        erase_filling = False
        erasing = False
        erase_pushed = False
        print("Eraser OFF")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if selected_id in mask_per_id:
            push_history()
            points_per_id[selected_id].clear()
            del mask_per_id[selected_id]
            rebuild_overlay()
            modified = True
            print(f"Cleared id={selected_id}")

# -------------------- Setup --------------------
cv2.namedWindow("SAM Label Tool", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SAM Label Tool", 1200, 800)
cv2.setMouseCallback("SAM Label Tool", mouse_callback)
load_mask()
print_image_id()  # Print initial image ID (1-based)

# -------------------- Main Loop --------------------
while True:
    combined = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0) if overlay_visible else image.copy()
    preview = combined.copy()

    if drawing and not manual_mode and time.time() - mouse_down_time >= CLICK_HOLD_THRESHOLD:
        manual_mode = True
        print("→ MANUAL DRAW mode")

    # Live outlines
    if drawing and manual_mode:
        # Brush outline
        cv2.circle(preview, last_mouse_pos, brush_radius, 1)
    if erasing:
        # Eraser outline (red)
        cv2.circle(preview, last_mouse_pos, eraser_radius, (0, 0, 255), 1)
    if filling:
        # Fill mode uses brush radius (show same outline)
        cv2.circle(preview, last_mouse_pos, brush_radius, 1)
    if erase_filling:
        # Erase-fill uses eraser radius (red outline)
        cv2.circle(preview, last_mouse_pos, eraser_radius, (0, 0, 255), 1)

    # Radius preview after +/- or wheel, when not actively drawing/erasing/fill
    now = time.time()
    if (
        now < radius_preview_expire and radius_preview_tool is not None and
        not drawing and not erasing and not filling and not erase_filling
    ):
        if radius_preview_tool == 'brush':
            cv2.circle(preview, last_mouse_pos, brush_radius, 1)
        elif radius_preview_tool == 'eraser':
            cv2.circle(preview, last_mouse_pos, eraser_radius, (0, 0, 255), 1)

    # Show SAM prompt points for current ID
    if selected_id in points_per_id:
        for (px, py) in points_per_id[selected_id]:
            cv2.circle(preview, (px, py), 4, (0, 255, 255), -1)

    cv2.imshow("SAM Label Tool", preview)
    key = cv2.waitKeyEx(10)

    if key == 27:  # ESC
        save_mask()
        break

    elif key in (ord('h'), ord('H')):
        overlay_visible = not overlay_visible
        print(f"Overlay {'ON' if overlay_visible else 'OFF'}")

    elif key in (ord('z'), ord('Z')):
        undo()

    elif 48 <= key <= 57:  # 0-9
        new_id = key - 48
        if new_id in id_to_color:
            selected_id = new_id
            selected_color = id_to_color[new_id]
            print(f"Selected id={selected_id}")

    # + / - to change radius of ACTIVE tool by 5 (clamped to [5, 50])
    elif key in (ord('+'), ord('=')):  # '=' is often the key you press with Shift to get '+'
        if erasing or erase_filling:
            eraser_radius = clamp(eraser_radius + 5)
            print(f"Eraser radius: {eraser_radius}")
            radius_preview_tool = 'eraser'
        else:
            brush_radius = clamp(brush_radius + 5)
            print(f"Brush radius: {brush_radius}")
            radius_preview_tool = 'brush'
        radius_preview_expire = time.time() + 0.7

    elif key in (ord('-'), ord('_')):
        if erasing or erase_filling:
            eraser_radius = clamp(eraser_radius - 5)
            print(f"Eraser radius: {eraser_radius}")
            radius_preview_tool = 'eraser'
        else:
            brush_radius = clamp(brush_radius - 5)
            print(f"Brush radius: {brush_radius}")
            radius_preview_tool = 'brush'
        radius_preview_expire = time.time() + 0.7

    elif key == ord('n'):
        goto_next_image()

    elif key == ord('b'):
        goto_prev_image()

    elif key == ord('/'):
        goto_next_unlabeled_image()

    elif key in (ord('c'), ord('C')):
        push_history()
        mask_per_id.clear()
        points_per_id.clear()
        mask_overlay[:] = 0
        modified = True
        print("Cleared ALL labels and SAM points for current image.")

    elif key in (ord('m'), ord('M')):
        # Use current cursor location as SAM prompt on the NEXT image
        x, y = last_mouse_pos
        print(f"Auto-label: using cursor ({x},{y}) on next image for id={selected_id}")

        # Move to next image (this will save current mask and load/reset next)
        goto_next_image()

        # Apply SAM on the newly loaded image at the same coordinate
        push_history()
        points_per_id[selected_id] = [(x, y)]
        print(f"Added SAM point ({x},{y}) for id={selected_id} on new image")
        run_sam(points_per_id[selected_id])
        modified = True

cv2.destroyAllWindows()
