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
                import ultralytics

                device = msg.get("device", None)
                model = ultralytics.YOLO(str(msg["model"]))  # type: ignore[attr-defined]
                sys.stdout.write(json.dumps({"type": "ready"}) + "\n")
                sys.stdout.flush()
            except Exception:
                sys.stdout.write(
                    json.dumps({"type": "ready", "ok": False, "error": traceback.format_exc()}) + "\n"
                )
                sys.stdout.flush()
            continue

        if mtype == "infer":
            req_id = msg.get("id")
            try:
                if model is None:
                    raise RuntimeError("worker not initialized")
                msg["device"] = device
                payload = _infer_one(msg, model)
                sys.stdout.write(json.dumps({"type": "result", "id": req_id, "ok": True, "payload": payload}) + "\n")
                sys.stdout.flush()
            except Exception:
                sys.stdout.write(
                    json.dumps({"type": "result", "id": req_id, "ok": False, "error": traceback.format_exc()})
                    + "\n"
                )
                sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
