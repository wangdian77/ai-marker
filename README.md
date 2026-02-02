# YOLO Pose Semi-Auto Labeler

Local, human-in-the-loop labeling tool for Ultralytics YOLO pose models.

It runs pose inference on a folder of images, shows an overlay (bbox + keypoints), lets you review multiple candidate variants, and then accept/save (or reject) each sample into a clean output dataset.

## Features

- GUI and CLI modes
- Load a local YOLO pose model (`.pt`, Ultralytics)
- Batch through an image folder; resume by skipping already-labeled outputs
- Candidate generation: multiple post-processing variants + optional jitter candidates for quick comparison
- Accept/Save into a new dataset (`output/images/` + `output/labels/`), or Reject (copy to rejected folder or hard delete)
- Built-in OpenCV editor to manually fix bbox/keypoints
- Optional Ultralytics `data.yaml` import for class names + pose settings (plus a few labeler-specific extensions)

## Install

Recommended: use a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Dependencies:

```bash
pip install ultralytics opencv-python numpy
```

Optional (recommended):

```bash
pip install PySide6  # GUI
pip install pyyaml   # load project YAML
pip install pillow   # nicer Chinese text rendering in OpenCV editor (Windows)
```

### Optional: GPU (CUDA) Support

If you have an NVIDIA GPU and want faster inference, install the CUDA build of PyTorch.

This project uses the same Python interpreter you run the GUI with (example below uses `C:/Python314/python.exe`).

```bash
# uninstall CPU wheels (if installed)
C:/Python314/python.exe -m pip uninstall -y torch torchvision

# install CUDA wheels (cu126)
C:/Python314/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# verify
C:/Python314/python.exe -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
```

Note: GPU support depends on your CUDA + PyTorch setup and your Ultralytics version.

## Output Format (YOLO Pose)

Each line = one instance:

```text
class x_center y_center w h kpt1x kpt1y kpt1v kpt2x kpt2y kpt2v ...
```

- All coordinates are normalized to `[0, 1]`
- `kptv` uses `0/2` toggling in this tool (0 = absent, 2 = visible)

## GUI (Recommended)

Start:

```bash
python yolo_labeler_gui.py
```

In the UI:

1. Select `Model (.pt)`
2. (Optional) Select `Project YAML` (e.g. `buff.yaml`) to load class names and pose settings
2. Select `Input Folder`
3. Select `Output Folder`
4. (Optional) Select `Rejected Folder`
5. Click `Load`

### Project YAML (Ultralytics data.yaml)

This tool can import an Ultralytics-style dataset YAML, for example:

```yaml
nc: 4
names:
  0: RR
  1: RW
  2: BR
  3: BW
kpt_shape: [5, 3]
```

It uses:

- `names` to show class names in the overlay
- `kpt_shape[0]` as the expected keypoint count; if mismatched, saving is blocked

Optional extensions supported by this tool (you may add them into your YAML):

```yaml
# Optional: names for keypoints (0..K-1)
kpt_names: ["p0", "p1", "p2", "p3", "p4"]

# Optional: skeleton edges (0-based indices)
skeleton:
  - [0, 1]
  - [1, 2]

# Optional: only keep these classes when generating candidates
allow_classes: [0, 2]

# Optional: set GUI defaults
labeler_defaults:
  conf: 0.25
  iou: 0.7
  candidates: 6
  kpt_jitter_px: 2.0
  bbox_jitter_px: 1.0
```

Controls:

- `Variant`: post-processing preset
- `Candidates`: how many candidates per image (candidate 0 is the base)
- `Kpt Jitter (px)`: keypoint gaussian jitter std-dev
- `BBox Jitter (px)`: whole-instance translation jitter std-dev
- `Seed` + `Re-roll`: regenerate random candidates
- `Pick Candidate`: choose which candidate to preview/save

Implementation note:

- In GUI mode, inference runs in a separate worker process (helps avoid Windows crashes from long-running inference in the UI process).

Device:

- `Device`: select `CPU` or `GPU0 (cuda:0)` (GPU requires CUDA PyTorch installed)

Actions:

- `Accept / Save`: writes to `output/images/` and `output/labels/`
- `Reject`: default copies to `rejected/` (does NOT modify inputs)
- `Edit`: opens an OpenCV editor window to manually fix bbox/keypoints
- `Clear`: clears all instances for the current candidate (saves an empty label if accepted)
- `Delete rejected input images (DANGEROUS)`: enables hard delete on reject
- `Skip existing output labels`: resume/skip images that already have output labels

## Shortcuts

These shortcuts work in the GUI:

- Next: `Right` / `D` / `Space`
- Prev: `Left` / `A`
- Save: `S`
- Reject: `R`
- Edit: `E`
- Clear: `Del` / `Backspace`
- Next Variant: `V`
- Next Candidate: `C`

## OpenCV Editor (Edit)

The editor is used to manually correct the auto-labeled results. Final saved labels are still standard YOLO pose labels.

Keys inside the editor window:

- Save and return: `s`
- Cancel: `Esc`
- Toggle help: `h`

Keypoints:

- Drag a keypoint: move it
- Right-click near a keypoint: toggle visibility `2 <-> 0`
- Select keypoint id: `0-9`
- Set selected keypoint at cursor: `n`
- Delete selected keypoint (set invisible): `x`

Lines (visual aid only, not saved to labels):

- Line mode toggle: `l` (click two times to create a line)
- Undo last line: `u`
- Line endpoint snapping: clicking near a keypoint snaps to it
- RGB color: use the `R/G/B` sliders in the editor window

Rotated bbox (visual aid only, not saved to labels):

- Select an instance by clicking inside its bbox, then rotate with `[` / `]`

Note:

- OpenCV does not render Chinese text well by default. This tool uses Pillow + Windows fonts (e.g. `C:\Windows\Fonts\msyh.ttc`) to render Chinese help text in the editor.

## CLI (OpenCV Window)

Basic:

```bash
python yolo_labeler.py --model path\\to\\model.pt --input path\\to\\images --output out_dataset
```

Use variants JSON:

```bash
python yolo_labeler.py --model model.pt --input images --output out_dataset --variants variants.example.json
```

Resume:

```bash
python yolo_labeler.py --model model.pt --input images --output out_dataset --skip-existing
```

Reject behavior:

- default: copy to `out_dataset/rejected/`
- destructive: `--delete-rejected`

```bash
python yolo_labeler.py --model model.pt --input images --output out_dataset --delete-rejected
```

CLI keys:

- `a`: accept/save
- `r`: reject
- `v`: switch variant
- `n` / `p`: next / previous
- `h`: toggle help overlay
- `q`: quit

Editing (CLI):

- left-drag on keypoint: move keypoint
- left-drag inside bbox: move the whole instance
- right-click on keypoint: toggle visibility (2 <-> 0)

## Variants

See `variants.example.json`.

Common fields:

- `min_area`: filter small boxes
- `expand`: expand bbox ratio
- `dedupe_iou`: IoU dedupe threshold
- `kpt_conf_ge`: keypoint confidence gate (below -> set `v=0`)

## Repo

- Issues/feature requests: use GitHub Issues
