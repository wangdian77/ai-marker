from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable when executed as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_train_worker import worker_main


if __name__ == "__main__":
    raise SystemExit(worker_main())
