# Data Parsing

Prepares and labels training data for all networks. Two tools are provided:
`mask_labeling.py` for interactive annotation and `mask_editing.py` for
batch conversion of existing datasets into the project's standard format.

---

## Installation

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Segment Anything Model (SAM) — required for mask_labeling.py only

SAM is installed from source:

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

Then download a SAM checkpoint:

```bash
# ViT-H (highest quality, ~2.5 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (~1.2 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (~375 MB, fastest)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Pass the checkpoint path to `mask_labeling.py` with the `-p` argument.
`mask_editing.py` does not require SAM or a GPU.

---

## mask_labeling.py — Interactive Annotation Tool

SAM-assisted labeling tool. Supports click-to-segment, manual paint/erase,
and flood-fill. Loads existing masks for refinement.

### Usage

```bash
python mask_labeling.py \
  -p /path/to/sam_checkpoint.pth \
  -i /path/to/input_images/ \
  -o /path/to/output_masks/ \
  -c /path/to/color_map.yaml
```

| Argument | Description |
|---|---|
| `-p` | SAM checkpoint path (e.g. `sam_vit_h_4b8939.pth`) |
| `-i` | Input images directory |
| `-o` | Output masks directory |
| `-c` | YAML color map file |
| `--no-fill` | Skip auto-filling small undefined holes |

### Color Map Format

```yaml
id_to_color_mapping:
  0: [61, 93, 255]    # non-drivable
  1: [0, 255, 220]    # drivable
```

Color maps for each task are in `config/color_map/`.

### Controls

| Input | Action |
|---|---|
| `0`–`9` | Select class ID to paint |
| Short left-click | SAM prompt — auto-segment region |
| Hold left-click | Paint brush |
| Middle-click | Erase brush |
| Ctrl + left-click | Flood-fill from cursor |
| `+` / `-` | Increase / decrease brush radius |
| `H` | Toggle mask overlay |
| `Z` | Undo |
| `N` / `B` | Next / previous image |
| `M` | Auto-label next unlabeled image |
| `C` | Clear all labels on current image |
| `/` | Jump to next unlabeled image |
| `ESC` | Save and exit |

### Output

One PNG mask per input image saved to the output directory, color-coded
according to the provided color map.

---

## mask_editing.py — Batch Dataset Preparation

Converts third-party datasets (Rellis-3D, GOOSE, CASSED, etc.) into the
standard directory structure expected by all training scripts. Remaps
source label colors/IDs to the project class IDs and copies resized images.

Uses [Hydra](https://hydra.cc/) for configuration.

### Usage

```bash
# Use defaults from config/config.yaml
python mask_editing.py

# Override dataset and color map
python mask_editing.py label_config=freespaceseg_goose color_map=freespaceseg
```

### Configuration

**`config/config.yaml`** — set the root path where datasets are stored:

```yaml
dataset_dir: /path/to/off-road_datasets/
defaults:
  - label_config: freespaceseg_rellis3d
  - color_map: freespaceseg
```

**`config/label_config/*.yaml`** — one file per source dataset:

```yaml
dataset: Rellis_3D
src_image_pattern: "**/*.jpg"
src_gt_mask_pattern: "**/*.png"
dst_image_dir: AORAS/FreespaceSeg/Rellis_3D/Rellis_3D/images/
dst_gt_mask_dir: AORAS/FreespaceSeg/Rellis_3D/Rellis_3D/gt_masks/
label_to_id_mapping:
  0: 0
  1: 1
color_thresh: 10
overwrite_existing: false
```

**`config/color_map/*.yaml`** — maps project class ID to output RGB color.
Available: `freespaceseg`, `terrainseg`, `objectseg`.

### Output Structure

```
{dataset_dir}/AORAS/{Task}/{Dataset}/{Dataset}/
├── images/       ← resized source images
└── gt_masks/     ← remapped color masks (PNG)
```

This structure is the required input for all training scripts.

### Supported Datasets

| label_config value | Dataset |
|---|---|
| `freespaceseg_rellis3d` | Rellis-3D |
| `freespaceseg_goose` | GOOSE |
| `freespaceseg_cassed` | CASSED |
| `freespaceseg_offsed` | OFFSED |
| `freespaceseg_yamahacmu` | Yamaha-CMU |
| `freespaceseg_orfd` | ORFD |
