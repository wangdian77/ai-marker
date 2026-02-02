# YOLO Labeler (Local Model + Human-in-the-loop)

Minimal interactive tool to:
- load a local trained YOLO model (Ultralytics)
- auto-label images (POSE: bbox + keypoints)
- show the labeled result to the user (bbox + keypoints)
- accept -> save to a new dataset (`images/` + `labels/`)
- reject -> move to `rejected/` (or hard delete with a flag)
- switch between multiple post-processing "variants" to improve dataset quality

## Install

```bash
pip install ultralytics opencv-python numpy
```

GUI mode (AnyLabeling-like buttons):

```bash
pip install PySide6
```

## Run

```bash
python yolo_labeler.py --model path/to/best.pt --input path/to/images --output out_dataset
```

GUI:

```bash
python yolo_labeler_gui.py
```

Use variants:

```bash
python yolo_labeler.py --model best.pt --input images --output out_dataset --variants variants.example.json
```

Resume/skip already labeled:

```bash
python yolo_labeler.py --model best.pt --input images --output out_dataset --skip-existing
```

Reject behavior:
- default: copy rejected images to `out_dataset/rejected/` (input is not modified)
- destructive option: `--delete-rejected`

## Keys

- `a`: accept/save
- `r`: reject
- `v`: switch variant
- `n` / `p`: next / previous
- `h`: toggle help overlay
- `q`: quit

Editing (current variant only):
- left-drag on a keypoint: move that keypoint
- left-drag inside a bbox: move the whole person (bbox + keypoints)
- right-click on a keypoint: toggle visibility (2 <-> 0)
- `0-9`: set current class id for new boxes
- `[` / `]`: decrement/increment current class id

## Output Format

- Images: `out_dataset/images/*.jpg|png|...`
- Labels: `out_dataset/labels/*.txt` in Ultralytics pose format (normalized):

`class x_center y_center w h kpt1x kpt1y kpt1v kpt2x kpt2y kpt2v ...`
