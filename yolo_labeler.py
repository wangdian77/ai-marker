import argparse
import json
import os
import shutil
import traceback
import unicodedata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import threading
import queue

import cv2
import numpy as np


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: Optional[float] = None

    def as_xyxy_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass
class Keypoint:
    x: float
    y: float
    v: int = 2  # 0=absent, 1=occluded, 2=visible (Ultralytics convention)
    conf: Optional[float] = None


@dataclass
class PoseInstance:
    bbox: Box
    kpts: List[Keypoint]


@dataclass
class LineSeg:
    x1: float
    y1: float
    x2: float
    y2: float
    bgr: Tuple[int, int, int]


def copy_pose_instances(poses: List[PoseInstance]) -> List[PoseInstance]:
    out: List[PoseInstance] = []
    for p in poses:
        b = p.bbox
        b2 = Box(b.x1, b.y1, b.x2, b.y2, cls=b.cls, conf=b.conf)
        k2 = [Keypoint(k.x, k.y, v=int(k.v), conf=k.conf) for k in p.kpts]
        out.append(PoseInstance(bbox=b2, kpts=k2))
    return out


def jitter_pose_instances(
    poses: List[PoseInstance],
    img_w: int,
    img_h: int,
    kpt_std_px: float,
    bbox_std_px: float,
    rng: np.random.Generator,
) -> List[PoseInstance]:
    if kpt_std_px <= 0 and bbox_std_px <= 0:
        return copy_pose_instances(poses)

    out = copy_pose_instances(poses)
    for p in out:
        if bbox_std_px > 0:
            dx = float(rng.normal(0.0, bbox_std_px))
            dy = float(rng.normal(0.0, bbox_std_px))
            p.bbox = clamp_box(
                Box(p.bbox.x1 + dx, p.bbox.y1 + dy, p.bbox.x2 + dx, p.bbox.y2 + dy, cls=p.bbox.cls, conf=p.bbox.conf),
                img_w,
                img_h,
            )
            p.kpts = [clamp_kpt(Keypoint(k.x + dx, k.y + dy, v=int(k.v), conf=k.conf), img_w, img_h) for k in p.kpts]

        if kpt_std_px > 0:
            for i, k in enumerate(p.kpts):
                if k.v <= 0:
                    continue
                dx = float(rng.normal(0.0, kpt_std_px))
                dy = float(rng.normal(0.0, kpt_std_px))
                p.kpts[i] = clamp_kpt(Keypoint(k.x + dx, k.y + dy, v=int(k.v), conf=k.conf), img_w, img_h)

    return out


def generate_pose_candidates(
    raw_poses: List[PoseInstance],
    img_w: int,
    img_h: int,
    variant: Dict[str, Any],
    num_candidates: int,
    seed: int,
    kpt_std_px: float,
    bbox_std_px: float,
) -> List[List[PoseInstance]]:
    base = apply_variant(raw_poses, img_w, img_h, variant)
    candidates: List[List[PoseInstance]] = []
    rng = np.random.default_rng(seed)

    n = max(1, int(num_candidates))
    # candidate[0] is always the base (no jitter) for easy comparison.
    candidates.append(copy_pose_instances(base))
    for _ in range(n - 1):
        candidates.append(jitter_pose_instances(base, img_w, img_h, kpt_std_px, bbox_std_px, rng))
    return candidates


def list_images(input_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    paths: List[Path] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)


def clamp_box(b: Box, w: int, h: int) -> Box:
    x1 = float(np.clip(b.x1, 0, w - 1))
    y1 = float(np.clip(b.y1, 0, h - 1))
    x2 = float(np.clip(b.x2, 0, w - 1))
    y2 = float(np.clip(b.y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, cls=b.cls, conf=b.conf)


def box_area(b: Box) -> float:
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = box_area(a) + box_area(b) - inter
    return 0.0 if ua <= 0 else inter / ua


def expand_box(b: Box, expand_ratio: float) -> Box:
    if expand_ratio <= 0:
        return b
    cx = (b.x1 + b.x2) / 2.0
    cy = (b.y1 + b.y2) / 2.0
    bw = (b.x2 - b.x1)
    bh = (b.y2 - b.y1)
    bw2 = bw * (1.0 + 2.0 * expand_ratio)
    bh2 = bh * (1.0 + 2.0 * expand_ratio)
    return Box(
        x1=cx - bw2 / 2.0,
        y1=cy - bh2 / 2.0,
        x2=cx + bw2 / 2.0,
        y2=cy + bh2 / 2.0,
        cls=b.cls,
        conf=b.conf,
    )


def dedupe_boxes(boxes: List[Box], iou_thresh: float) -> List[Box]:
    if iou_thresh <= 0:
        return boxes
    # Prefer higher confidence; unknown conf goes last.
    def key(b: Box) -> float:
        return float(b.conf) if b.conf is not None else -1.0

    kept: List[Box] = []
    for b in sorted(boxes, key=key, reverse=True):
        if all(iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def to_yolo_pose_line(p: PoseInstance, img_w: int, img_h: int) -> str:
    b = p.bbox
    x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    parts = [
        str(b.cls),
        f"{xc / img_w:.6f}",
        f"{yc / img_h:.6f}",
        f"{bw / img_w:.6f}",
        f"{bh / img_h:.6f}",
    ]
    for kp in p.kpts:
        parts.append(f"{kp.x / img_w:.6f}")
        parts.append(f"{kp.y / img_h:.6f}")
        parts.append(str(int(kp.v)))
    return " ".join(parts)


def draw_overlay(
    img_bgr: np.ndarray,
    poses: List[PoseInstance],
    class_names: Dict[int, str],
    show_conf: bool,
    title_lines: List[str],
    current_class: int,
    help_on: bool,
    kpt_names: Optional[List[str]] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    show_kpt_labels: bool = False,
) -> np.ndarray:
    out = img_bgr.copy()

    for p in poses:
        b = p.bbox
        x1, y1, x2, y2 = b.as_xyxy_int()
        color = (0, 255, 0) if b.cls == current_class else (0, 180, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = class_names.get(b.cls, str(b.cls))
        if show_conf and b.conf is not None:
            label = f"{label} {b.conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        visible_xy: Dict[int, Tuple[int, int]] = {}
        for i, kp in enumerate(p.kpts):
            if kp.v <= 0:
                continue
            px, py = int(round(kp.x)), int(round(kp.y))
            visible_xy[i] = (px, py)
            cv2.circle(out, (px, py), 4, (255, 255, 0), -1)
            if show_kpt_labels:
                kp_label = str(i)
                if kpt_names is not None and 0 <= i < len(kpt_names) and kpt_names[i]:
                    kp_label = kpt_names[i]
                cv2.putText(out, kp_label, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        if skeleton:
            for a_i, b_i in skeleton:
                if a_i in visible_xy and b_i in visible_xy:
                    cv2.line(out, visible_xy[a_i], visible_xy[b_i], (255, 255, 0), 2)
        else:
            # Fallback: connect consecutive visible keypoints.
            pts = [visible_xy[i] for i in sorted(visible_xy.keys())]
            for a, b2 in zip(pts, pts[1:]):
                cv2.line(out, a, b2, (255, 255, 0), 2)

    y = 18
    for line in title_lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        y += 22

    if help_on:
        help_lines = [
            "Keys: a=accept/save  r=reject  n=next  p=prev  v=variant  q=quit  h=help",
            "Edit: left-drag keypoint=move | left-drag inside bbox=move person | right-click keypoint=toggle vis",
            "Class: 0-9 set current class | [ and ] cycle",
        ]
        y2 = out.shape[0] - 10 - 22 * (len(help_lines) - 1)
        for line in help_lines:
            cv2.putText(out, line, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
            y2 += 22
    return out


def point_in_box(x: int, y: int, b: Box) -> bool:
    return b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2


def clamp_kpt(kp: Keypoint, w: int, h: int) -> Keypoint:
    x = float(np.clip(kp.x, 0, w - 1))
    y = float(np.clip(kp.y, 0, h - 1))
    return Keypoint(x=x, y=y, v=int(kp.v), conf=kp.conf)


def nearest_kpt(poses: List[PoseInstance], x: int, y: int, radius_px: int) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    best_d2 = float(radius_px * radius_px)
    for pi, p in enumerate(poses):
        for ki, kp in enumerate(p.kpts):
            if kp.v <= 0:
                continue
            d2 = float((kp.x - x) ** 2 + (kp.y - y) ** 2)
            if d2 <= best_d2:
                best_d2 = d2
                best = (pi, ki)
    return best


class Editor:
    def __init__(self) -> None:
        self.dragging: bool = False
        self.mode: Optional[str] = None  # "move_kpt" | "move_pose" | None
        self.start: Tuple[int, int] = (0, 0)
        self.last: Tuple[int, int] = (0, 0)
        self.active_pose_idx: Optional[int] = None
        self.active_kpt_idx: Optional[int] = None

    def reset(self) -> None:
        self.dragging = False
        self.mode = None
        self.active_pose_idx = None
        self.active_kpt_idx = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_variants(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return [
            {"name": "default", "min_area": 0, "expand": 0.0, "dedupe_iou": 0.0, "kpt_conf_ge": None, "round": True},
            {"name": "strict", "min_area": 800, "expand": 0.0, "dedupe_iou": 0.6, "kpt_conf_ge": 0.35, "round": True},
            {"name": "loose", "min_area": 200, "expand": 0.02, "dedupe_iou": 0.7, "kpt_conf_ge": 0.15, "round": True},
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("variants file must be a JSON array")
    for v in data:
        if "name" not in v:
            raise ValueError("each variant must include 'name'")
    return data


def apply_variant(poses: List[PoseInstance], img_w: int, img_h: int, v: Dict[str, Any]) -> List[PoseInstance]:
    min_area = float(v.get("min_area", 0))
    expand = float(v.get("expand", 0.0))
    dedupe_iou = float(v.get("dedupe_iou", 0.0))
    keep_conf_ge = v.get("keep_conf_ge", None)
    kpt_conf_ge = v.get("kpt_conf_ge", None)
    allow_classes = v.get("allow_classes", None)
    do_round = bool(v.get("round", True))

    out: List[PoseInstance] = []
    for p in poses:
        b = p.bbox
        if allow_classes is not None and isinstance(allow_classes, list):
            if int(b.cls) not in [int(x) for x in allow_classes]:
                continue
        if keep_conf_ge is not None and b.conf is not None:
            if float(b.conf) < float(keep_conf_ge):
                continue

        b2 = expand_box(b, expand)
        b2 = clamp_box(b2, img_w, img_h)
        if box_area(b2) < min_area:
            continue

        kpts2: List[Keypoint] = []
        for kp in p.kpts:
            kp2 = clamp_kpt(kp, img_w, img_h)
            if kpt_conf_ge is not None and kp2.conf is not None:
                if float(kp2.conf) < float(kpt_conf_ge):
                    kp2 = Keypoint(x=kp2.x, y=kp2.y, v=0, conf=kp2.conf)
            kpts2.append(kp2)

        if do_round:
            x1, y1, x2, y2 = b2.as_xyxy_int()
            b2 = Box(float(x1), float(y1), float(x2), float(y2), cls=b2.cls, conf=b2.conf)
            kpts2 = [Keypoint(float(int(round(k.x))), float(int(round(k.y))), v=int(k.v), conf=k.conf) for k in kpts2]

        out.append(PoseInstance(bbox=b2, kpts=kpts2))

    # Dedupe based on bbox IoU
    if dedupe_iou > 0:
        boxes_only = [p.bbox for p in out]
        kept_boxes = dedupe_boxes(boxes_only, dedupe_iou)
        kept_set = {(b.x1, b.y1, b.x2, b.y2, b.cls, b.conf) for b in kept_boxes}
        out = [p for p in out if (p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2, p.bbox.cls, p.bbox.conf) in kept_set]

    return out


def read_class_names(model: Any, classes_file: Optional[Path]) -> Dict[int, str]:
    if classes_file is not None:
        names: Dict[int, str] = {}
        for i, line in enumerate(classes_file.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            names[i] = line
        return names

    model_names = getattr(model, "names", None)
    if isinstance(model_names, dict):
        return {int(k): str(v) for k, v in model_names.items()}
    if isinstance(model_names, list):
        return {i: str(n) for i, n in enumerate(model_names)}
    return {}


def infer_pose_ultralytics(
    model: Any,
    img_bgr: np.ndarray,
    conf: float,
    iou_th: float,
    device: Optional[str] = None,
) -> List[PoseInstance]:
    kwargs: Dict[str, Any] = {"source": img_bgr, "conf": conf, "iou": iou_th, "verbose": False}
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results:
        return []
    r0 = results[0]
    bxs = getattr(r0, "boxes", None)
    kps = getattr(r0, "keypoints", None)
    if bxs is None or kps is None:
        return []

    xyxy = bxs.xyxy.cpu().numpy() if hasattr(bxs.xyxy, "cpu") else np.asarray(bxs.xyxy)
    cls = bxs.cls.cpu().numpy() if hasattr(bxs.cls, "cpu") else np.asarray(bxs.cls)
    cf = bxs.conf.cpu().numpy() if hasattr(bxs.conf, "cpu") else np.asarray(bxs.conf)

    kxy = None
    if hasattr(kps, "xy"):
        kxy = kps.xy.cpu().numpy() if hasattr(kps.xy, "cpu") else np.asarray(kps.xy)
    kconf = None
    if hasattr(kps, "conf"):
        kconf = kps.conf.cpu().numpy() if hasattr(kps.conf, "cpu") else np.asarray(kps.conf)

    out: List[PoseInstance] = []
    n = min(len(xyxy), len(cls), len(cf), len(kxy) if kxy is not None else 0)
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i]
        b = Box(float(x1), float(y1), float(x2), float(y2), cls=int(cls[i]), conf=float(cf[i]))
        kpts: List[Keypoint] = []
        if kxy is not None:
            for j, (x, y) in enumerate(kxy[i]):
                kc = float(kconf[i][j]) if kconf is not None else None
                # Default v=2 for predicted keypoints.
                kpts.append(Keypoint(x=float(x), y=float(y), v=2, conf=kc))
        out.append(PoseInstance(bbox=b, kpts=kpts))
    return out


def safe_reject(src: Path, rejected_dir: Path, hard_delete: bool) -> None:
    if hard_delete:
        src.unlink(missing_ok=True)
        return
    ensure_dir(rejected_dir)
    dst = rejected_dir / src.name
    if dst.exists():
        stem = src.stem
        dst = rejected_dir / f"{stem}_{src.stat().st_mtime_ns}{src.suffix}"
    shutil.copy2(str(src), str(dst))


def save_sample(
    src_img: Path,
    img_bgr: np.ndarray,
    poses: List[PoseInstance],
    out_images: Path,
    out_labels: Path,
) -> None:
    ensure_dir(out_images)
    ensure_dir(out_labels)

    dst_img = out_images / src_img.name
    if dst_img.exists():
        stem = src_img.stem
        dst_img = out_images / f"{stem}_{src_img.stat().st_mtime_ns}{src_img.suffix}"

    shutil.copy2(str(src_img), str(dst_img))

    label_path = out_labels / f"{dst_img.stem}.txt"
    h, w = img_bgr.shape[:2]
    lines = [to_yolo_pose_line(p, w, h) for p in poses]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLO model file (.pt)")
    ap.add_argument("--input", required=True, help="Input images directory")
    ap.add_argument("--output", required=True, help="Output dataset directory")
    ap.add_argument("--rejected-dir", default=None, help="Rejected images dir (default: <output>/rejected)")
    ap.add_argument("--delete-rejected", action="store_true", help="Hard delete rejected images")
    ap.add_argument("--conf", type=float, default=0.25, help="Predict confidence")
    ap.add_argument("--iou", type=float, default=0.7, help="Predict NMS IoU")
    ap.add_argument("--variants", default=None, help="Variants JSON file")
    ap.add_argument("--skip-existing", action="store_true", help="Skip images with existing output label")
    ap.add_argument("--classes", default=None, help="Optional classes.txt (one name per line)")
    args = ap.parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rejected_dir = Path(args.rejected_dir) if args.rejected_dir else (output_dir / "rejected")
    variants_path = Path(args.variants) if args.variants else None
    classes_file = Path(args.classes) if args.classes else None

    try:
        import ultralytics
    except Exception:
        print("Missing dependency: ultralytics. Install: pip install ultralytics")
        return 2

    YOLOCls = getattr(ultralytics, "YOLO")
    model = YOLOCls(str(model_path))
    class_names = read_class_names(model, classes_file)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = list_images(input_dir, exts)
    if not images:
        print(f"No images found under: {input_dir}")
        return 1

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    ensure_dir(output_dir)

    variants = load_variants(variants_path)
    variant_idx = 0
    show_conf = True
    help_on = True

    current_class = 0
    editor = Editor()
    window = "YOLO Labeler"

    state: Dict[str, Any] = {
        "img": None,
        "img_path": None,
        "raw_poses": None,
        "variant_poses": None,
    }

    def recompute_variant_boxes() -> None:
        img = state["img"]
        raw = state["raw_poses"]
        if img is None or raw is None:
            return
        h, w = img.shape[:2]
        v = variants[variant_idx]
        state["variant_poses"] = apply_variant(raw, w, h, v)

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        del flags, userdata
        img = state["img"]
        poses: Optional[List[PoseInstance]] = state.get("variant_poses")
        if img is None or poses is None:
            return

        h, w = img.shape[:2]
        hit = nearest_kpt(poses, x, y, radius_px=10)

        if event == cv2.EVENT_LBUTTONDOWN:
            editor.dragging = True
            editor.start = (x, y)
            editor.last = (x, y)
            editor.active_pose_idx = None
            editor.active_kpt_idx = None
            if hit is not None:
                editor.active_pose_idx, editor.active_kpt_idx = hit
                editor.mode = "move_kpt"
            else:
                # If click inside a bbox, move the whole pose.
                for i, p in enumerate(poses):
                    if point_in_box(x, y, p.bbox):
                        editor.active_pose_idx = i
                        editor.mode = "move_pose"
                        break

        elif event == cv2.EVENT_MOUSEMOVE and editor.dragging:
            dx = x - editor.last[0]
            dy = y - editor.last[1]
            if editor.mode == "move_kpt" and editor.active_pose_idx is not None and editor.active_kpt_idx is not None:
                p = poses[editor.active_pose_idx]
                kp = p.kpts[editor.active_kpt_idx]
                kp2 = Keypoint(x=kp.x + dx, y=kp.y + dy, v=int(kp.v), conf=kp.conf)
                kp2 = clamp_kpt(kp2, w, h)
                p.kpts[editor.active_kpt_idx] = kp2
            elif editor.mode == "move_pose" and editor.active_pose_idx is not None:
                p = poses[editor.active_pose_idx]
                b = p.bbox
                b2 = Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, cls=b.cls, conf=b.conf)
                b2 = clamp_box(b2, w, h)
                p.bbox = b2
                p.kpts = [clamp_kpt(Keypoint(k.x + dx, k.y + dy, v=int(k.v), conf=k.conf), w, h) for k in p.kpts]
            editor.last = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and editor.dragging:
            editor.reset()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Toggle keypoint visibility under cursor (2 <-> 0)
            if hit is not None:
                pi, ki = hit
                kp = poses[pi].kpts[ki]
                poses[pi].kpts[ki] = Keypoint(kp.x, kp.y, v=(0 if kp.v > 0 else 2), conf=kp.conf)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        # Resume behavior: if output label exists, skip.
        if args.skip_existing:
            candidate_label = out_labels / f"{img_path.stem}.txt"
            if candidate_label.exists():
                idx += 1
                continue

        state["img"] = img
        state["img_path"] = img_path
        state["raw_poses"] = infer_pose_ultralytics(model, img, conf=float(args.conf), iou_th=float(args.iou))
        recompute_variant_boxes()
        editor.reset()

        while True:
            v = variants[variant_idx]
            poses = list(state.get("variant_poses") or [])

            title = [
                f"{idx + 1}/{len(images)}  {img_path.name}",
                f"variant={v.get('name','?')}  instances={len(state.get('variant_poses') or [])}  class={current_class} ({class_names.get(current_class, str(current_class))})",
                f"predict(conf={args.conf:.2f}, iou={args.iou:.2f})  reject={'DELETE' if args.delete_rejected else 'COPY'}",
            ]
            vis = draw_overlay(img, poses, class_names, show_conf, title, current_class, help_on)
            cv2.imshow(window, vis)

            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue

            ch = chr(k) if 32 <= k < 127 else ""
            if ch == "q":
                cv2.destroyAllWindows()
                return 0
            if ch == "h":
                help_on = not help_on
                continue
            if ch == "c":
                show_conf = not show_conf
                continue
            if ch == "v":
                variant_idx = (variant_idx + 1) % len(variants)
                recompute_variant_boxes()
                editor.reset()
                continue
            if ch == "a":
                save_sample(img_path, img, state.get("variant_poses") or [], out_images, out_labels)
                idx += 1
                break
            if ch == "r":
                safe_reject(img_path, rejected_dir, hard_delete=bool(args.delete_rejected))
                idx += 1
                break
            if ch == "n":
                idx += 1
                break
            if ch == "p":
                idx -= 1
                break
            if ch == "[":
                current_class = max(0, current_class - 1)
                continue
            if ch == "]":
                current_class = current_class + 1
                continue
            if ch.isdigit():
                current_class = int(ch)
                continue

        # continue outer loop

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def edit_poses_opencv(
    img_bgr: np.ndarray,
    poses: List[PoseInstance],
    class_names: Optional[Dict[int, str]] = None,
    kpt_names: Optional[List[str]] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    lang: str = "en",
) -> Tuple[bool, List[PoseInstance]]:
    """Blocking OpenCV editor for a single image.

    - Drag near a keypoint to move it.
    - Drag inside bbox to move the whole instance.
    - Right-click near a keypoint to toggle visibility (2 <-> 0).
    - Keypoint add/delete: select kpt id with 0-9, 'n' to set it at cursor, 'x' to delete (v=0).
    - Line drawing: 'l' to start/finish a line (two clicks); RGB sliders control line color.
    - Connect points: same as line drawing; lines are visual aids (not saved to labels).
    - Rotate bbox (visual aid): select an instance then '['/']' adjust angle.
    - Press 's' to accept changes, 'ESC' to cancel.
    """
    if class_names is None:
        class_names = {}

    img = img_bgr
    h, w = img.shape[:2]
    current = copy_pose_instances(poses)
    editor = Editor()
    window = "Edit Pose"
    help_on = True
    ui_lang = (lang or "en").lower()

    def _find_cjk_font_path() -> Optional[str]:
        # Windows default fonts (best-effort)
        candidates = [
            r"C:\\Windows\\Fonts\\msyh.ttc",  # Microsoft YaHei
            r"C:\\Windows\\Fonts\\msyhbd.ttc",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simsun.ttc",
        ]
        for p in candidates:
            if Path(p).exists():
                return p
        return None

    _cjk_font_path = _find_cjk_font_path()

    def _has_non_ascii(s: str) -> bool:
        return any(ord(ch) > 127 for ch in s)

    def _draw_text(canvas_bgr: np.ndarray, text: str, x: int, y: int) -> None:
        # Use PIL for non-ASCII text to avoid OpenCV garbling.
        if _has_non_ascii(text) and _cjk_font_path is not None:
            try:
                from PIL import Image, ImageDraw, ImageFont

                # Convert BGR -> RGB for PIL
                rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(_cjk_font_path, 18)

                # Outline + fill
                ox = 2
                for dx in (-ox, 0, ox):
                    for dy in (-ox, 0, ox):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
                draw.text((x, y), text, font=font, fill=(255, 255, 0))

                canvas_bgr[:] = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
                return
            except Exception:
                # Fall back to OpenCV below
                pass

        # ASCII fallback (OpenCV)
        cv2.putText(canvas_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(canvas_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    # Visual aids (not saved)
    lines: List[LineSeg] = []
    line_start: Optional[Tuple[int, int]] = None
    mode: Optional[str] = None  # None | "line"
    last_mouse: Tuple[int, int] = (w // 2, h // 2)
    selected_pose: Optional[int] = None
    selected_kpt: int = 0
    angles: Dict[int, float] = {}

    # RGB color sliders (line color) integrated into main window.
    def current_bgr() -> Tuple[int, int, int]:
        try:
            r = int(cv2.getTrackbarPos("R", window))
            g = int(cv2.getTrackbarPos("G", window))
            b = int(cv2.getTrackbarPos("B", window))
            return (b, g, r)
        except Exception:
            return (0, 0, 255)

    def get_screen_size() -> Tuple[int, int]:
        # Best-effort screen size for initial fit.
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return 1280, 720

    def get_viewport() -> Tuple[int, int]:
        # Return current window image rect if available.
        try:
            _, _, ww, wh = cv2.getWindowImageRect(window)
            if ww > 0 and wh > 0:
                return int(ww), int(wh)
        except Exception:
            pass
        sw, sh = get_screen_size()
        return min(1400, int(sw * 0.92)), min(900, int(sh * 0.86))

    def map_view_to_img(x: int, y: int, scale: float, off_x: int, off_y: int) -> Optional[Tuple[int, int]]:
        ix = int(round((x - off_x) / scale))
        iy = int(round((y - off_y) / scale))
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    def hit_kpt(x: int, y: int, scale: float) -> Optional[Tuple[int, int]]:
        # Keep similar visual hit radius regardless of zoom.
        r = int(round(10.0 / max(0.2, scale)))
        r = int(np.clip(r, 4, 30))
        return nearest_kpt(current, x, y, radius_px=r)

    view_state = {"scale": 1.0, "off_x": 0, "off_y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        del flags, userdata
        nonlocal line_start, mode, last_mouse, selected_pose

        scale = float(view_state.get("scale", 1.0))
        off_x = int(view_state.get("off_x", 0))
        off_y = int(view_state.get("off_y", 0))
        mapped = map_view_to_img(x, y, scale, off_x, off_y)
        if mapped is None:
            return
        ix, iy = mapped
        last_mouse = (ix, iy)

        if mode == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                # Snap to nearest visible keypoint if close.
                snap = hit_kpt(ix, iy, scale)
                if snap is not None:
                    pi, ki = snap
                    try:
                        sx = int(round(current[pi].kpts[ki].x))
                        sy = int(round(current[pi].kpts[ki].y))
                        ix, iy = sx, sy
                    except Exception:
                        pass

                if line_start is None:
                    line_start = (ix, iy)
                else:
                    x1, y1 = line_start
                    lines.append(LineSeg(float(x1), float(y1), float(ix), float(iy), bgr=current_bgr()))
                    line_start = None
            return

        hit = hit_kpt(ix, iy, scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            editor.dragging = True
            editor.last = (ix, iy)
            editor.active_pose_idx = None
            editor.active_kpt_idx = None
            if hit is not None:
                editor.active_pose_idx, editor.active_kpt_idx = hit
                selected_pose = editor.active_pose_idx
                selected_kpt = editor.active_kpt_idx
                editor.mode = "move_kpt"
            else:
                for i, p in enumerate(current):
                    if point_in_box(ix, iy, p.bbox):
                        editor.active_pose_idx = i
                        selected_pose = i
                        editor.mode = "move_pose"
                        break

        elif event == cv2.EVENT_MOUSEMOVE and editor.dragging:
            dx = ix - editor.last[0]
            dy = iy - editor.last[1]
            if editor.mode == "move_kpt" and editor.active_pose_idx is not None and editor.active_kpt_idx is not None:
                p = current[editor.active_pose_idx]
                kp = p.kpts[editor.active_kpt_idx]
                kp2 = clamp_kpt(Keypoint(kp.x + dx, kp.y + dy, v=int(kp.v), conf=kp.conf), w, h)
                p.kpts[editor.active_kpt_idx] = kp2
            elif editor.mode == "move_pose" and editor.active_pose_idx is not None:
                p = current[editor.active_pose_idx]
                b = p.bbox
                b2 = clamp_box(Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, cls=b.cls, conf=b.conf), w, h)
                p.bbox = b2
                p.kpts = [clamp_kpt(Keypoint(k.x + dx, k.y + dy, v=int(k.v), conf=k.conf), w, h) for k in p.kpts]
            editor.last = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP and editor.dragging:
            editor.reset()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if hit is not None:
                pi, ki = hit
                kp = current[pi].kpts[ki]
                current[pi].kpts[ki] = Keypoint(kp.x, kp.y, v=(0 if kp.v > 0 else 2), conf=kp.conf)
                selected_pose = pi
                selected_kpt = ki

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        # Trackbars on main window.
        try:
            cv2.createTrackbar("R", window, 255, 255, lambda v: None)
            cv2.createTrackbar("G", window, 0, 255, lambda v: None)
            cv2.createTrackbar("B", window, 0, 255, lambda v: None)
        except Exception:
            pass
        # Best-effort initial size.
        vw, vh = get_viewport()
        try:
            cv2.resizeWindow(window, vw, vh)
        except Exception:
            pass
        cv2.setMouseCallback(window, on_mouse)
        while True:
            # Allow closing by the window [X]
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return False, poses
            except Exception:
                pass

            vw, vh = get_viewport()
            scale = min(float(vw) / float(w), float(vh) / float(h))
            scale = float(np.clip(scale, 0.1, 6.0))
            scaled_w = int(round(w * scale))
            scaled_h = int(round(h * scale))
            off_x = max(0, (vw - scaled_w) // 2)
            off_y = max(0, (vh - scaled_h) // 2)
            view_state["scale"] = scale
            view_state["off_x"] = off_x
            view_state["off_y"] = off_y

            # Compose overlay
            mode_txt = "LINE" if mode == "line" else "MOVE"
            if ui_lang.startswith("zh"):
                title = [
                    f"模式: {'画线' if mode == 'line' else '移动'}  点={selected_kpt}",
                    "按键: s=保存  ESC/q=取消  h=帮助  l=画线  u=撤销线  t=中/英",
                    "按键2: 0-9=点编号  n=在光标处设置点  x=隐藏点  [ ]=旋转框(仅显示)",
                ]
            else:
                title = [
                    f"Mode: {mode_txt}  kpt={selected_kpt}",
                    "Keys: s=save  ESC/q=cancel  h=help  l=line  u=undo line  t=lang",
                    "Keys2: 0-9=kpt id  n=set kpt at cursor  x=hide kpt  [ ]=rotate bbox (visual)",
                ]
            vis = draw_overlay(
                img,
                current,
                class_names,
                False,
                title,
                current_class=0,
                help_on=help_on,
                kpt_names=kpt_names,
                skeleton=skeleton,
                show_kpt_labels=True,
            )

            # Scale to viewport for display.
            vis_small = cv2.resize(vis, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            canvas = np.zeros((vh, vw, 3), dtype=np.uint8)
            canvas[off_y : off_y + scaled_h, off_x : off_x + scaled_w] = vis_small

            # Draw lines (visual aids) on canvas (scaled)
            for ln in lines:
                x1 = int(round(off_x + ln.x1 * scale))
                y1 = int(round(off_y + ln.y1 * scale))
                x2 = int(round(off_x + ln.x2 * scale))
                y2 = int(round(off_y + ln.y2 * scale))
                cv2.line(canvas, (x1, y1), (x2, y2), ln.bgr, 2)
            if line_start is not None:
                x1 = int(round(off_x + line_start[0] * scale))
                y1 = int(round(off_y + line_start[1] * scale))
                x2 = int(round(off_x + last_mouse[0] * scale))
                y2 = int(round(off_y + last_mouse[1] * scale))
                cv2.line(canvas, (x1, y1), (x2, y2), current_bgr(), 2)

            # Draw rotated bbox visual aid on canvas (scaled)
            if selected_pose is not None and 0 <= selected_pose < len(current):
                ang = float(angles.get(selected_pose, 0.0))
                b = current[selected_pose].bbox
                cx = (b.x1 + b.x2) / 2.0
                cy = (b.y1 + b.y2) / 2.0
                bw = max(1.0, b.x2 - b.x1)
                bh = max(1.0, b.y2 - b.y1)
                rect = ((float(cx) * scale + off_x, float(cy) * scale + off_y), (float(bw) * scale, float(bh) * scale), float(ang))
                pts = cv2.boxPoints(rect)
                pts = np.int32(pts)
                cv2.polylines(canvas, [pts], True, (255, 0, 255), 2)

            if help_on:
                if ui_lang.startswith("zh"):
                    help_lines = [
                        "鼠标: 拖动关键点/框 | 右键点: 可见性 2<->0",
                        "画线: l 开关, 点击两点(自动吸附关键点), u 撤销, R/G/B 滑条调色",
                        "关键点: 0-9 选编号, n 在光标处设置, x 隐藏, [ ] 旋转框(仅显示)",
                        "退出: s 保存 | ESC/q 取消 | 关闭窗口 | t 切换中英文",
                    ]
                else:
                    help_lines = [
                        "Mouse: drag keypoint/bbox | Right-click keypoint: toggle vis 2<->0",
                        "Line: l toggle, click 2 points (snaps to keypoints), u undo, R/G/B sliders",
                        "Keypoints: 0-9 select, n set at cursor, x hide, [ ] rotate bbox (visual)",
                        "Exit: s save | ESC/q cancel | close window | t toggle language",
                    ]
                y0 = 18
                for hl in help_lines:
                    _draw_text(canvas, hl, 10, y0)
                    y0 += 22

            cv2.imshow(window, canvas)
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k == 27:  # ESC
                cv2.destroyWindow(window)
                return False, poses

            ch = chr(k) if 32 <= k < 127 else ""
            if ch == "h":
                help_on = not help_on
            elif ch == "t":
                ui_lang = "en" if ui_lang.startswith("zh") else "zh"
            elif ch == "s":
                cv2.destroyWindow(window)
                return True, current
            elif ch == "q":
                cv2.destroyWindow(window)
                return False, poses
            elif ch == "l":
                mode = None if mode == "line" else "line"
                line_start = None
            elif ch == "u":
                if lines:
                    lines.pop()
            elif ch == "[":
                if selected_pose is not None:
                    angles[selected_pose] = float(angles.get(selected_pose, 0.0)) - 5.0
            elif ch == "]":
                if selected_pose is not None:
                    angles[selected_pose] = float(angles.get(selected_pose, 0.0)) + 5.0
            elif ch == "n":
                # Add/set selected keypoint at cursor
                if selected_pose is not None and 0 <= selected_pose < len(current):
                    p = current[selected_pose]
                    if 0 <= selected_kpt < len(p.kpts):
                        x0, y0 = last_mouse
                        kp = p.kpts[selected_kpt]
                        p.kpts[selected_kpt] = Keypoint(float(x0), float(y0), v=2, conf=kp.conf)
            elif ch == "x":
                # Delete selected keypoint (set invisible)
                if selected_pose is not None and 0 <= selected_pose < len(current):
                    p = current[selected_pose]
                    if 0 <= selected_kpt < len(p.kpts):
                        kp = p.kpts[selected_kpt]
                        p.kpts[selected_kpt] = Keypoint(kp.x, kp.y, v=0, conf=kp.conf)
            elif ch.isdigit():
                selected_kpt = int(ch)
    except Exception:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
        raise RuntimeError("OpenCV editor failed:\n" + traceback.format_exc())


@dataclass
class EditorUpdate:
    image_bgr: np.ndarray
    poses: List[PoseInstance]
    class_names: Dict[int, str]
    kpt_names: Optional[List[str]]
    skeleton: Optional[List[Tuple[int, int]]]
    lang: str


@dataclass
class EditorResult:
    ok: bool
    poses: List[PoseInstance]
    closed: bool = False


class PersistentPoseEditor:
    """Keep one OpenCV editor window open.

    - GUI can push new images/poses via `send_update()`.
    - User edits current content.
    - When user presses 's', a result is emitted via `result_queue`.
    - When user presses 'q'/ESC/closes window, editor stops.
    """

    def __init__(self) -> None:
        self.update_queue: "queue.Queue[EditorUpdate]" = queue.Queue()
        self.result_queue: "queue.Queue[EditorResult]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def send_update(self, upd: EditorUpdate) -> None:
        # Keep only the latest update.
        try:
            while True:
                self.update_queue.get_nowait()
        except Exception:
            pass
        self.update_queue.put(upd)

    def _run(self) -> None:
        import cv2
        import numpy as np

        window = "Edit Pose"
        help_on = True

        # Per-image state
        img = None
        current: List[PoseInstance] = []
        class_names: Dict[int, str] = {}
        kpt_names: Optional[List[str]] = None
        skeleton: Optional[List[Tuple[int, int]]] = None
        ui_lang = "en"

        lines: List[LineSeg] = []
        line_start: Optional[Tuple[int, int]] = None
        mode: Optional[str] = None  # None | "line"
        last_mouse: Tuple[int, int] = (0, 0)
        selected_pose: Optional[int] = None
        selected_kpt: int = 0
        angles: Dict[int, float] = {}

        editor = Editor()
        view_state = {"scale": 1.0, "off_x": 0, "off_y": 0, "w": 1, "h": 1, "zoom": 1.0}
        delete_confirm_idx: Optional[int] = None
        delete_confirm_time: float = 0.0

        def _find_cjk_font_path() -> Optional[str]:
            candidates = [
                r"C:\\Windows\\Fonts\\msyh.ttc",
                r"C:\\Windows\\Fonts\\msyhbd.ttc",
                r"C:\\Windows\\Fonts\\simhei.ttf",
                r"C:\\Windows\\Fonts\\simsun.ttc",
            ]
            for p in candidates:
                if Path(p).exists():
                    return p
            return None

        _cjk_font_path = _find_cjk_font_path()

        def _has_non_ascii(s: str) -> bool:
            return any(ord(ch) > 127 for ch in s)

        def _draw_text(canvas_bgr: np.ndarray, text: str, x: int, y: int) -> None:
            if _has_non_ascii(text) and _cjk_font_path is not None:
                try:
                    from PIL import Image, ImageDraw, ImageFont

                    rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb)
                    draw = ImageDraw.Draw(img_pil)
                    font = ImageFont.truetype(_cjk_font_path, 18)
                    ox = 2
                    for dx in (-ox, 0, ox):
                        for dy in (-ox, 0, ox):
                            if dx == 0 and dy == 0:
                                continue
                            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
                    draw.text((x, y), text, font=font, fill=(255, 255, 0))
                    canvas_bgr[:] = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
                    return
                except Exception:
                    pass
            cv2.putText(canvas_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
            cv2.putText(canvas_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        def current_bgr() -> Tuple[int, int, int]:
            try:
                r = int(cv2.getTrackbarPos("R", window))
                g = int(cv2.getTrackbarPos("G", window))
                b = int(cv2.getTrackbarPos("B", window))
                return (b, g, r)
            except Exception:
                return (0, 0, 255)

        def get_screen_size() -> Tuple[int, int]:
            try:
                import ctypes

                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
            except Exception:
                return 1280, 720

        def get_viewport() -> Tuple[int, int]:
            try:
                _, _, ww, wh = cv2.getWindowImageRect(window)
                if ww > 0 and wh > 0:
                    return int(ww), int(wh)
            except Exception:
                pass
            sw, sh = get_screen_size()
            return min(1400, int(sw * 0.92)), min(900, int(sh * 0.86))

        def map_view_to_img(x: int, y: int, scale: float, off_x: int, off_y: int, iw: int, ih: int) -> Optional[Tuple[int, int]]:
            ix = int(round((x - off_x) / scale))
            iy = int(round((y - off_y) / scale))
            if 0 <= ix < iw and 0 <= iy < ih:
                return ix, iy
            return None

        def hit_kpt(x: int, y: int, scale: float) -> Optional[Tuple[int, int]]:
            r = int(round(10.0 / max(0.2, scale)))
            r = int(np.clip(r, 4, 30))
            return nearest_kpt(current, x, y, radius_px=r)

        def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
            del userdata
            nonlocal line_start, mode, last_mouse, selected_pose, selected_kpt
            if img is None:
                return

            if event == cv2.EVENT_MOUSEWHEEL:
                is_ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
                if is_ctrl:
                    z = float(view_state.get("zoom", 1.0))
                    if flags > 0:
                        z *= 1.1
                    else:
                        z /= 1.1
                    view_state["zoom"] = float(np.clip(z, 0.1, 10.0))
                return

            scale = float(view_state.get("scale", 1.0))
            off_x = int(view_state.get("off_x", 0))
            off_y = int(view_state.get("off_y", 0))
            iw = int(view_state.get("w", 1))
            ih = int(view_state.get("h", 1))
            mapped = map_view_to_img(x, y, scale, off_x, off_y, iw, ih)
            if mapped is None:
                return
            ix, iy = mapped
            last_mouse = (ix, iy)

            if mode == "line":
                if event == cv2.EVENT_LBUTTONDOWN:
                    snap = hit_kpt(ix, iy, scale)
                    if snap is not None:
                        pi, ki = snap
                        ix = int(round(current[pi].kpts[ki].x))
                        iy = int(round(current[pi].kpts[ki].y))
                    if line_start is None:
                        line_start = (ix, iy)
                    else:
                        x1, y1 = line_start
                        lines.append(LineSeg(float(x1), float(y1), float(ix), float(iy), bgr=current_bgr()))
                        line_start = None
                return

            hit = hit_kpt(ix, iy, scale)
            if event == cv2.EVENT_LBUTTONDOWN:
                editor.dragging = True
                editor.last = (ix, iy)
                editor.active_pose_idx = None
                editor.active_kpt_idx = None
                if hit is not None:
                    editor.active_pose_idx, editor.active_kpt_idx = hit
                    selected_pose = editor.active_pose_idx
                    selected_kpt = editor.active_kpt_idx
                    editor.mode = "move_kpt"
                else:
                    for i, p in enumerate(current):
                        if point_in_box(ix, iy, p.bbox):
                            editor.active_pose_idx = i
                            selected_pose = i
                            editor.mode = "move_pose"
                            break
            elif event == cv2.EVENT_MOUSEMOVE and editor.dragging:
                dx = ix - editor.last[0]
                dy = iy - editor.last[1]
                if editor.mode == "move_kpt" and editor.active_pose_idx is not None and editor.active_kpt_idx is not None:
                    p = current[editor.active_pose_idx]
                    kp = p.kpts[editor.active_kpt_idx]
                    p.kpts[editor.active_kpt_idx] = clamp_kpt(Keypoint(kp.x + dx, kp.y + dy, v=int(kp.v), conf=kp.conf), iw, ih)
                elif editor.mode == "move_pose" and editor.active_pose_idx is not None:
                    p = current[editor.active_pose_idx]
                    b = p.bbox
                    p.bbox = clamp_box(Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, cls=b.cls, conf=b.conf), iw, ih)
                    p.kpts = [clamp_kpt(Keypoint(k.x + dx, k.y + dy, v=int(k.v), conf=k.conf), iw, ih) for k in p.kpts]
                editor.last = (ix, iy)
            elif event == cv2.EVENT_LBUTTONUP and editor.dragging:
                editor.reset()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if hit is not None:
                    pi, ki = hit
                    kp = current[pi].kpts[ki]
                    current[pi].kpts[ki] = Keypoint(kp.x, kp.y, v=(0 if kp.v > 0 else 2), conf=kp.conf)
                    selected_pose = pi
                    selected_kpt = ki

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            cv2.createTrackbar("R", window, 255, 255, lambda v: None)
            cv2.createTrackbar("G", window, 0, 255, lambda v: None)
            cv2.createTrackbar("B", window, 0, 255, lambda v: None)
        except Exception:
            pass

        vw, vh = get_viewport()
        try:
            cv2.resizeWindow(window, vw, vh)
        except Exception:
            pass

        cv2.setMouseCallback(window, on_mouse)

        while not self._stop.is_set():
            # Pull latest update if any
            try:
                upd = self.update_queue.get_nowait()
                img = upd.image_bgr
                current = copy_pose_instances(upd.poses)
                class_names = dict(upd.class_names)
                kpt_names = upd.kpt_names
                skeleton = upd.skeleton
                ui_lang = (upd.lang or "en").lower()
                lines = []
                line_start = None
                mode = None
                selected_pose = None
                selected_kpt = 0
                angles = {}
                editor.reset()
            except Exception:
                pass

            if img is None:
                # Idle until first update
                cv2.waitKey(30)
                continue

            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    self.result_queue.put(EditorResult(ok=False, poses=[], closed=True))
                    return
            except Exception:
                pass

            ih, iw = img.shape[:2]
            view_state["w"] = iw
            view_state["h"] = ih

            vw, vh = get_viewport()
            base_scale = min(float(vw) / float(iw), float(vh) / float(ih))
            user_zoom = float(view_state.get("zoom", 1.0))
            scale = float(np.clip(base_scale * user_zoom, 0.05, 20.0))
            scaled_w = int(round(iw * scale))
            scaled_h = int(round(ih * scale))
            off_x = max(0, (vw - scaled_w) // 2)
            off_y = max(0, (vh - scaled_h) // 2)
            view_state["scale"] = scale
            view_state["off_x"] = off_x
            view_state["off_y"] = off_y

            mode_txt = "LINE" if mode == "line" else "MOVE"
            if ui_lang.startswith("zh"):
                title = [
                    f"模式: {'画线' if mode == 'line' else '移动'}  点={selected_kpt}",
                    "按键: s=应用  ESC/q=退出编辑  h=帮助  l=画线  u=撤销线  t=中/英",
                    "按键2: 0-9=点编号  n=设置点  x=隐藏点  [ ]=旋转框(仅显示)",
                ]
            else:
                title = [
                    f"Mode: {mode_txt}  kpt={selected_kpt}",
                    "Keys: s=apply  ESC/q=exit  h=help  l=line  u=undo  t=lang",
                    "Keys2: 0-9=kpt id  n=set kpt  x=hide kpt  [ ]=rotate bbox (visual)",
                ]

            vis = draw_overlay(
                img,
                current,
                class_names,
                False,
                title,
                current_class=0,
                help_on=False,
                kpt_names=kpt_names,
                skeleton=skeleton,
                show_kpt_labels=True,
            )

            vis_small = cv2.resize(vis, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            canvas = np.zeros((vh, vw, 3), dtype=np.uint8)
            canvas[off_y : off_y + scaled_h, off_x : off_x + scaled_w] = vis_small

            for ln in lines:
                x1 = int(round(off_x + ln.x1 * scale))
                y1 = int(round(off_y + ln.y1 * scale))
                x2 = int(round(off_x + ln.x2 * scale))
                y2 = int(round(off_y + ln.y2 * scale))
                cv2.line(canvas, (x1, y1), (x2, y2), ln.bgr, 2)
            if line_start is not None:
                x1 = int(round(off_x + line_start[0] * scale))
                y1 = int(round(off_y + line_start[1] * scale))
                x2 = int(round(off_x + last_mouse[0] * scale))
                y2 = int(round(off_y + last_mouse[1] * scale))
                cv2.line(canvas, (x1, y1), (x2, y2), current_bgr(), 2)

            if selected_pose is not None and 0 <= selected_pose < len(current):
                ang = float(angles.get(selected_pose, 0.0))
                b = current[selected_pose].bbox
                cx = (b.x1 + b.x2) / 2.0
                cy = (b.y1 + b.y2) / 2.0
                bw = max(1.0, b.x2 - b.x1)
                bh = max(1.0, b.y2 - b.y1)
                rect = ((float(cx) * scale + off_x, float(cy) * scale + off_y), (float(bw) * scale, float(bh) * scale), float(ang))
                pts = cv2.boxPoints(rect)
                pts = np.int32(pts)
                cv2.polylines(canvas, [pts], True, (255, 0, 255), 2)
                if delete_confirm_idx == selected_pose:
                    _draw_text(canvas, "Press DEL again to confirm delete", int(cx * scale + off_x), int(cy * scale + off_y - 20))

            if help_on:
                if ui_lang.startswith("zh"):
                    help_lines = [
                        "鼠标: 拖动点/框 | 右键点: 2<->0 | Ctrl+滚轮: 缩放",
                        "画线: l, 点两次(吸附点), u 撤销, R/G/B 滑条",
                        "点: 0-9 选, n 设置, x 隐藏, [ ] 旋转框(显示)",
                        "删除选中实例: 按两下 Delete/Backspace",
                        "编辑会保持打开, ESC/q 退出编辑窗口",
                    ]
                else:
                    help_lines = [
                        "Mouse: drag kpt/bbox | Right-click: 2<->0 | Ctrl+Wheel: Zoom",
                        "Line: l, 2 clicks (snap), u undo, R/G/B sliders",
                        "Kpt: 0-9 select, n set, x hide, [ ] rotate (visual)",
                        "Delete instance: press Delete/Backspace twice",
                        "Editor stays open; ESC/q to exit",
                    ]
                y0 = 18
                for hl in help_lines:
                    _draw_text(canvas, hl, 10, y0)
                    y0 += 22

            cv2.imshow(window, canvas)
            rk = cv2.waitKey(20)
            k = rk & 0xFF
            if k == 255 and rk == -1:
                continue

            is_del = (k == 8 or k == 255 or rk == 3014656 or rk == 2162688)
            
            if is_del:
                if selected_pose is not None:
                    now = time.time()
                    if delete_confirm_idx == selected_pose and (now - delete_confirm_time) < 2.0:
                        current.pop(selected_pose)
                        selected_pose = None
                        delete_confirm_idx = None
                    else:
                        delete_confirm_idx = selected_pose
                        delete_confirm_time = now
                continue
            
            if k != 255:
                if not is_del:
                    delete_confirm_idx = None

            if k == 27:
                self.result_queue.put(EditorResult(ok=False, poses=[], closed=True))
                return
            ch = chr(k) if 32 <= k < 127 else ""
            if ch == "h":
                help_on = not help_on
            elif ch == "t":
                ui_lang = "en" if ui_lang.startswith("zh") else "zh"
            elif ch == "q":
                self.result_queue.put(EditorResult(ok=False, poses=[], closed=True))
                return
            elif ch == "s":
                # Apply current edits to GUI (do not close)
                self.result_queue.put(EditorResult(ok=True, poses=copy_pose_instances(current), closed=False))
            elif ch == "l":
                mode = None if mode == "line" else "line"
                line_start = None
            elif ch == "u":
                if lines:
                    lines.pop()
            elif ch == "[":
                if selected_pose is not None:
                    angles[selected_pose] = float(angles.get(selected_pose, 0.0)) - 5.0
            elif ch == "]":
                if selected_pose is not None:
                    angles[selected_pose] = float(angles.get(selected_pose, 0.0)) + 5.0
            elif ch == "n":
                if selected_pose is not None and 0 <= selected_pose < len(current):
                    p = current[selected_pose]
                    if 0 <= selected_kpt < len(p.kpts):
                        x0, y0 = last_mouse
                        kp = p.kpts[selected_kpt]
                        p.kpts[selected_kpt] = Keypoint(float(x0), float(y0), v=2, conf=kp.conf)
            elif ch == "x":
                if selected_pose is not None and 0 <= selected_pose < len(current):
                    p = current[selected_pose]
                    if 0 <= selected_kpt < len(p.kpts):
                        kp = p.kpts[selected_kpt]
                        p.kpts[selected_kpt] = Keypoint(kp.x, kp.y, v=0, conf=kp.conf)
            elif ch.isdigit():
                selected_kpt = int(ch)
