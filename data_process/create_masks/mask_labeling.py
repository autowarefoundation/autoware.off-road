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
BRUSH_MIN, BRUSH_MAX = 5, 50    # updated min/max
brush_radius = 20               # initial radius (within [5, 50])

# Fill small undefined holes (in unlabeled regions) narrower than this many pixels
FILL_HOLE_WIDTH = 30  # Morphological closing kernel width (in pixels)

# -------------------- Load Color Config --------------------
with open(args.color_map, "r") as f:
    config = yaml.safe_load(f)
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

drawing = False
manual_mode = False
manual_pushed = False
erasing = False
erase_pushed = False
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
print("  • Middle Button Drag: Erase mask (red brush outline)")
print("  • Right Click: Clear mask for the current color ID")
print("  • Ctrl + Mouse Wheel: Adjust brush/eraser radius (fine control)")
print()
print("Keyboard:")
print("  • 0–9: Switch color ID manually")
print("  • + / -: Increase / decrease brush radius by 5 px (min 5, max 50)")
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

# -------------------- Mouse Callback --------------------
def mouse_callback(event, x, y, flags, param):
    global drawing, manual_mode, manual_pushed, erasing, erase_pushed
    global mouse_down_time, last_mouse_pos, modified, brush_radius
    last_mouse_pos = (x, y)

    if event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
        delta = 1 if flags > 0 else -1
        brush_radius = clamp(brush_radius + delta * 2)
        print(f"Brush radius: {brush_radius}")
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        manual_mode = False
        manual_pushed = False
        mouse_down_time = time.time()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and manual_mode:
            if not manual_pushed:
                push_history()
                manual_pushed = True
            tmp = np.zeros((h, w), np.uint8)
            cv2.circle(tmp, (x, y), brush_radius, 1, -1)
            area = tmp.astype(bool)
            mask_per_id.setdefault(selected_id, np.zeros((h, w), bool))[area] = True
            mask_overlay[area] = selected_color
            modified = True
        elif erasing:
            if not erase_pushed:
                push_history()
                erase_pushed = True
            tmp = np.zeros((h, w), np.uint8)
            cv2.circle(tmp, (x, y), brush_radius, 1, -1)
            area = tmp.astype(bool)
            for cid in list(mask_per_id.keys()):
                mask_per_id[cid][area] = False
                if not np.any(mask_per_id[cid]):
                    del mask_per_id[cid]
            mask_overlay[area] = 0
            modified = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        elapsed = time.time() - mouse_down_time
        if elapsed < CLICK_HOLD_THRESHOLD:
            push_history()
            points_per_id[selected_id].append((x, y))
            print(f"Added SAM point ({x},{y}) for id={selected_id}")
            run_sam(points_per_id[selected_id])
            modified = True
        else:
            manual_mode = False
            manual_pushed = False

    elif event == cv2.EVENT_MBUTTONDOWN:
        erasing = True
        erase_pushed = False
        print("Eraser ON")
    elif event == cv2.EVENT_MBUTTONUP:
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

    if drawing and manual_mode:
        cv2.circle(preview, last_mouse_pos, brush_radius, 1)
    if erasing:
        cv2.circle(preview, last_mouse_pos, brush_radius, (0, 0, 255), 1)

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

    # + / - to change brush size by 5 (clamped to [5, 50])
    elif key in (ord('+'), ord('=')):  # '=' is often the key you press with Shift to get '+'
        brush_radius = clamp(brush_radius + 5)
        print(f"Brush radius: {brush_radius}")
    elif key in (ord('-'), ord('_')):
        brush_radius = clamp(brush_radius - 5)
        print(f"Brush radius: {brush_radius}")

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
