import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict


def worker_main() -> int:
    import sys

    emit_lock = threading.Lock()

    def _emit(msg: Dict[str, Any]) -> None:
        with emit_lock:
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
        if mtype != "train":
            continue

        req_id = msg.get("id")
        try:
            os.environ.setdefault("YOLO_UPDATE_CHECK", "False")
            os.environ.setdefault("YOLO_ANALYTICS", "False")
            os.environ.setdefault("YOLO_VERBOSE", "False")

            _emit({"type": "status", "id": req_id, "phase": "import_ultralytics"})
            import ultralytics

            _emit({"type": "status", "id": req_id, "phase": "loading_model"})
            model = ultralytics.YOLO(str(msg["model"]))  # type: ignore[attr-defined]

            kwargs: Dict[str, Any] = {
                "data": str(msg["data"]),
                "epochs": int(msg.get("epochs", 100)),
                "imgsz": int(msg.get("imgsz", 640)),
                "batch": int(msg.get("batch", 16)),
                "device": msg.get("device", "cpu"),
                "project": str(msg.get("project", "train_runs")),
                "name": str(msg.get("name", "auto")),
                "verbose": False,
            }

            Path(str(kwargs["project"])).mkdir(parents=True, exist_ok=True)

            _emit({"type": "status", "id": req_id, "phase": "training"})
            progress_stop = threading.Event()

            def _progress_loop() -> None:
                value = 0
                while not progress_stop.wait(1.0):
                    if value < 95:
                        value += 1
                    _emit({"type": "progress", "id": req_id, "value": value})

            progress_thread = threading.Thread(target=_progress_loop, daemon=True)
            progress_thread.start()
            try:
                result = model.train(**kwargs)
            finally:
                progress_stop.set()
                progress_thread.join(timeout=0.3)

            save_dir = ""
            best = ""
            try:
                save_dir = str(getattr(result, "save_dir", ""))
            except Exception:
                save_dir = ""
            try:
                best = str(getattr(result, "best", ""))
            except Exception:
                best = ""

            _emit(
                {
                    "type": "result",
                    "id": req_id,
                    "ok": True,
                    "payload": {
                        "save_dir": save_dir,
                        "best": best,
                    },
                }
            )
        except Exception:
            _emit(
                {
                    "type": "result",
                    "id": req_id,
                    "ok": False,
                    "error": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
