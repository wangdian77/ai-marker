import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pose_to_dict(p: Any) -> Dict[str, Any]:
    b = p.bbox
    return {
        "cls": int(b.cls),
        "conf": (None if b.conf is None else float(b.conf)),
        "bbox": [float(b.x1), float(b.y1), float(b.x2), float(b.y2)],
        "kpts": [
            [float(k.x), float(k.y), int(k.v), (None if k.conf is None else float(k.conf))] for k in p.kpts
        ],
    }


def _infer_one(req: Dict[str, Any], model: Any) -> Dict[str, Any]:
    import cv2
    import torch

    from yolo_labeler import generate_pose_candidates, infer_pose_ultralytics

    cv2.setNumThreads(0)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    image_path = Path(req["image"])
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    variant = json.loads(req.get("variant_json", "{}"))
    conf = float(req.get("conf", 0.25))
    iou = float(req.get("iou", 0.7))
    device = req.get("device", None)

    raw = infer_pose_ultralytics(model, img, conf=conf, iou_th=iou, device=device)
    h, w = img.shape[:2]
    candidates = generate_pose_candidates(
        raw,
        img_w=w,
        img_h=h,
        variant=variant,
        num_candidates=int(req.get("num_candidates", 6)),
        seed=int(req.get("seed", 0)),
        kpt_std_px=float(req.get("kpt_std_px", 0.0)),
        bbox_std_px=float(req.get("bbox_std_px", 0.0)),
    )

    return {
        "image": str(image_path),
        "shape": [int(h), int(w)],
        "candidates": [[_pose_to_dict(p) for p in cand] for cand in candidates],
    }


def _infer_region(req: Dict[str, Any], model: Any) -> Dict[str, Any]:
    import cv2
    import numpy as np

    from yolo_labeler import Box, Keypoint, PoseInstance, clamp_box, clamp_kpt, infer_pose_ultralytics, iou

    image_path = Path(req["image"])
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    raw_box = req.get("bbox", None)
    if not (isinstance(raw_box, list) and len(raw_box) == 4):
        raise ValueError("infer_region requires bbox=[x1,y1,x2,y2]")

    x1, y1, x2, y2 = [float(v) for v in raw_box]
    target = clamp_box(Box(x1, y1, x2, y2, cls=0, conf=None), w, h)

    expand = float(req.get("expand", 0.2))
    if expand < 0:
        expand = 0.0

    # Expand in pixel space.
    bw = max(1.0, target.x2 - target.x1)
    bh = max(1.0, target.y2 - target.y1)
    ex1 = target.x1 - bw * expand
    ey1 = target.y1 - bh * expand
    ex2 = target.x2 + bw * expand
    ey2 = target.y2 + bh * expand
    exp_box = clamp_box(Box(ex1, ey1, ex2, ey2, cls=0, conf=None), w, h)

    ix1, iy1, ix2, iy2 = exp_box.as_xyxy_int()
    if ix2 <= ix1 or iy2 <= iy1:
        raise RuntimeError("infer_region: empty crop")

    crop = img[iy1:iy2, ix1:ix2]
    conf = float(req.get("conf", 0.25))
    iou_th = float(req.get("iou", 0.7))
    device = req.get("device", None)
    expected_kpts = req.get("expected_kpts", None)
    expected_k = int(expected_kpts) if expected_kpts is not None else None

    poses = infer_pose_ultralytics(model, crop, conf=conf, iou_th=iou_th, device=device)

    def to_full(p: PoseInstance) -> PoseInstance:
        b = p.bbox
        b2 = Box(b.x1 + ix1, b.y1 + iy1, b.x2 + ix1, b.y2 + iy1, cls=int(b.cls), conf=b.conf)
        kpts2 = []
        for k in p.kpts:
            k2 = clamp_kpt(Keypoint(k.x + ix1, k.y + iy1, v=int(k.v), conf=k.conf), w, h)
            kpts2.append(k2)
        if expected_k is not None:
            if len(kpts2) < expected_k:
                kpts2 = kpts2 + [Keypoint(0.0, 0.0, v=0, conf=None) for _ in range(expected_k - len(kpts2))]
            elif len(kpts2) > expected_k:
                kpts2 = kpts2[:expected_k]
        return PoseInstance(bbox=clamp_box(b2, w, h), kpts=kpts2)

    full = [to_full(p) for p in poses]
    best: PoseInstance | None = None
    best_score = -1.0
    for p in full:
        score = float(iou(Box(p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2, cls=0, conf=None), target))
        if score > best_score:
            best_score = score
            best = p

    # Fallback: if nothing detected in crop, return empty pose with requested bbox.
    if best is None:
        kpts: list[Keypoint] = []
        if expected_k is None:
            expected_k = 5
        cx = (target.x1 + target.x2) / 2.0
        cy = (target.y1 + target.y2) / 2.0
        if expected_k == 5:
            pts = [(cx, cy), (target.x1, target.y1), (target.x2, target.y1), (target.x1, target.y2), (target.x2, target.y2)]
            for px, py in pts:
                kpts.append(Keypoint(float(px), float(py), v=2, conf=None))
        else:
            cols = int(np.ceil(np.sqrt(expected_k)))
            rows = int(np.ceil(expected_k / cols))
            for i in range(expected_k):
                r = i // cols
                c = i % cols
                px = target.x1 + (c + 0.5) * ((target.x2 - target.x1) / cols)
                py = target.y1 + (r + 0.5) * ((target.y2 - target.y1) / rows)
                kpts.append(Keypoint(float(px), float(py), v=2, conf=None))
        best = PoseInstance(bbox=Box(target.x1, target.y1, target.x2, target.y2, cls=0, conf=1.0), kpts=kpts)

    return {
        "image": str(image_path),
        "shape": [int(h), int(w)],
        "bbox": [float(target.x1), float(target.y1), float(target.x2), float(target.y2)],
        "pose": _pose_to_dict(best),
        "score": float(best_score),
    }


def worker_main() -> int:
    """JSONL protocol on stdin/stdout.

    Input lines:
      {"type":"init","model":"...","device":"cpu"|"cuda:0"}
      {"type":"infer", ...}
      {"type":"quit"}

    Output lines:
      {"type":"ready"}
      {"type":"result","id":...,"ok":true,"payload":{...}}
      {"type":"result","id":...,"ok":false,"error":"traceback..."}
    """
    import sys

    model = None
    device = None

    def _emit(msg: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except Exception:
            continue

        mtype = msg.get("type")
        if mtype == "quit":
            return 0

        if mtype == "init":
            try:
                import os

                # Ultralytics may perform update/analytics checks during import.
                # In restricted/slow network environments this can stall import for tens of seconds.
                os.environ.setdefault("YOLO_UPDATE_CHECK", "False")
                os.environ.setdefault("YOLO_ANALYTICS", "False")
                os.environ.setdefault("YOLO_VERBOSE", "False")

                _emit({"type": "status", "phase": "import_ultralytics"})
                import ultralytics

                device = msg.get("device", None)
                _emit({"type": "status", "phase": "loading_model"})
                model = ultralytics.YOLO(str(msg["model"]))  # type: ignore[attr-defined]
                _emit({"type": "ready"})
            except Exception:
                _emit({"type": "ready", "ok": False, "error": traceback.format_exc()})
            continue

        if mtype == "infer":
            req_id = msg.get("id")
            try:
                if model is None:
                    raise RuntimeError("worker not initialized")
                msg["device"] = device
                _emit({"type": "status", "phase": "loading_image", "id": req_id})
                payload = _infer_one(msg, model)
                _emit({"type": "result", "kind": "infer", "id": req_id, "ok": True, "payload": payload})
            except Exception:
                _emit({"type": "result", "kind": "infer", "id": req_id, "ok": False, "error": traceback.format_exc()})

        if mtype == "infer_region":
            req_id = msg.get("id")
            try:
                if model is None:
                    raise RuntimeError("worker not initialized")
                msg["device"] = device
                _emit({"type": "status", "phase": "loading_image", "id": req_id})
                payload = _infer_region(msg, model)
                _emit({"type": "result", "kind": "infer_region", "id": req_id, "ok": True, "payload": payload})
            except Exception:
                _emit({"type": "result", "kind": "infer_region", "id": req_id, "ok": False, "error": traceback.format_exc()})

    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
