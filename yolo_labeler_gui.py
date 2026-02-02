import os
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from PySide6 import QtCore, QtGui, QtWidgets

    _HAS_QT = True
except Exception:  # pragma: no cover
    _HAS_QT = False


@dataclass
class SessionConfig:
    model_path: Path
    input_dir: Path
    output_dir: Path
    rejected_dir: Path
    conf: float
    iou: float
    skip_existing: bool
    delete_rejected: bool


def _parse_names(names: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names, list):
        for i, v in enumerate(names):
            out[int(i)] = str(v)
    return out


def load_project_yaml(path: Path) -> Dict[str, Any]:
    """Load Ultralytics-like data.yaml and optional extensions.

    Supported base keys (like buff.yaml):
    - names: {id: name} or [name]
    - kpt_shape: [K, 3]

    Optional extensions (for this tool):
    - kpt_names: [name0, name1, ...]
    - skeleton: [[a,b], [a,b], ...] (0-based indices)
    - allow_classes: [0,1,...]
    - labeler_defaults: {conf: 0.25, iou: 0.7, candidates: 6, kpt_jitter_px: 2.0, bbox_jitter_px: 1.0}
    """
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Missing dependency: pyyaml (pip install pyyaml)") from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("project yaml must be a mapping/dict")

    names = _parse_names(data.get("names", {}))

    kpt_shape = data.get("kpt_shape", None)
    expected_kpts: Optional[int] = None
    if isinstance(kpt_shape, list) and len(kpt_shape) >= 1:
        try:
            expected_kpts = int(kpt_shape[0])
        except Exception:
            expected_kpts = None

    kpt_names = data.get("kpt_names", None)
    if kpt_names is not None and not isinstance(kpt_names, list):
        raise ValueError("kpt_names must be a list")

    skeleton_raw = data.get("skeleton", None)
    skeleton: Optional[List[List[int]]] = None
    if skeleton_raw is not None:
        if not isinstance(skeleton_raw, list):
            raise ValueError("skeleton must be a list")
        skeleton = []
        for e in skeleton_raw:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                skeleton.append([int(e[0]), int(e[1])])

    allow_classes_raw = data.get("allow_classes", None)
    allow_classes: Optional[List[int]] = None
    if allow_classes_raw is not None:
        if not isinstance(allow_classes_raw, list):
            raise ValueError("allow_classes must be a list")
        allow_classes = [int(x) for x in allow_classes_raw]

    defaults = data.get("labeler_defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("labeler_defaults must be a dict")

    return {
        "path": str(path),
        "names": names,
        "expected_kpts": expected_kpts,
        "kpt_names": kpt_names,
        "skeleton": skeleton,
        "allow_classes": allow_classes,
        "defaults": defaults,
    }


def list_images(input_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths: List[Path] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)


if _HAS_QT:

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._lang = "en"  # "en" | "zh"

            self.setWindowTitle("YOLO Pose Labeler")
            self.resize(1280, 800)

            self._model_path: Optional[Path] = None
            self._input_dir: Optional[Path] = None
            self._output_dir: Optional[Path] = None
            self._rejected_dir: Optional[Path] = None

            self._images: List[Path] = []
            self._idx: int = 0

            self._img_bgr = None
            self._candidates: Optional[List[List[Any]]] = None
            self._busy: bool = False
            self._pix_orig: Optional[QtGui.QPixmap] = None

            # Persistent OpenCV editor session (optional)
            self._edit_mode = False
            self._editor = None
            self._editor_timer = QtCore.QTimer(self)
            self._editor_timer.setInterval(100)
            self._editor_timer.timeout.connect(self._poll_editor)

            self._project: Optional[Dict[str, Any]] = None
            self._class_names: Dict[int, str] = {}
            self._kpt_names: Optional[List[str]] = None
            self._skeleton: Optional[List[List[int]]] = None
            self._expected_kpts: Optional[int] = None
            self._allow_classes: Optional[List[int]] = None

            # Inference worker process (avoids Windows heap corruption 0xc0000374)
            self._worker = QtCore.QProcess(self)
            self._worker.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
            self._worker.readyReadStandardOutput.connect(self._on_worker_stdout)
            self._worker.readyReadStandardError.connect(self._on_worker_stderr)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker_buf_out = ""
            self._worker_buf_err = ""
            self._worker_ready = False
            self._inflight_id: Optional[int] = None
            self._inflight_path: Optional[str] = None
            self._next_req_id = 1
            self._pending_reload = False
            self._restart_requested = False

            left = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left)
            form = QtWidgets.QFormLayout()
            left_layout.addLayout(form)

            # Language toggle
            self.lang_btn = QtWidgets.QPushButton("中文")
            self.lang_btn.clicked.connect(self._toggle_lang)
            left_layout.addWidget(self.lang_btn)

            # Project YAML
            self.project_yaml_edit = QtWidgets.QLineEdit()
            self.project_yaml_btn = QtWidgets.QPushButton("Browse")
            self.project_yaml_btn.clicked.connect(self._pick_project_yaml)
            self.project_yaml_label = QtWidgets.QLabel("Project YAML")
            form.addRow(self.project_yaml_label, self._hbox(self.project_yaml_edit, self.project_yaml_btn))

            self.model_edit = QtWidgets.QLineEdit()
            self.model_btn = QtWidgets.QPushButton("Browse")
            self.model_btn.clicked.connect(self._pick_model)
            self.model_label = QtWidgets.QLabel("Model (.pt)")
            form.addRow(self.model_label, self._hbox(self.model_edit, self.model_btn))

            self.input_edit = QtWidgets.QLineEdit()
            self.input_btn = QtWidgets.QPushButton("Browse")
            self.input_btn.clicked.connect(self._pick_input)
            self.input_label = QtWidgets.QLabel("Input Folder")
            form.addRow(self.input_label, self._hbox(self.input_edit, self.input_btn))

            self.output_edit = QtWidgets.QLineEdit()
            self.output_btn = QtWidgets.QPushButton("Browse")
            self.output_btn.clicked.connect(self._pick_output)
            self.output_label = QtWidgets.QLabel("Output Folder")
            form.addRow(self.output_label, self._hbox(self.output_edit, self.output_btn))

            self.rejected_edit = QtWidgets.QLineEdit()
            self.rejected_btn = QtWidgets.QPushButton("Browse")
            self.rejected_btn.clicked.connect(self._pick_rejected)
            self.rejected_label = QtWidgets.QLabel("Rejected Folder")
            form.addRow(self.rejected_label, self._hbox(self.rejected_edit, self.rejected_btn))

            self.conf_spin = QtWidgets.QDoubleSpinBox()
            self.conf_spin.setRange(0.0, 1.0)
            self.conf_spin.setDecimals(2)
            self.conf_spin.setSingleStep(0.05)
            self.conf_spin.setValue(0.25)
            self.conf_label = QtWidgets.QLabel("Conf")
            form.addRow(self.conf_label, self.conf_spin)

            self.iou_spin = QtWidgets.QDoubleSpinBox()
            self.iou_spin.setRange(0.0, 1.0)
            self.iou_spin.setDecimals(2)
            self.iou_spin.setSingleStep(0.05)
            self.iou_spin.setValue(0.70)
            self.iou_label = QtWidgets.QLabel("IoU")
            form.addRow(self.iou_label, self.iou_spin)

            self.skip_chk = QtWidgets.QCheckBox("Skip existing output labels")
            self.skip_chk.setChecked(True)
            left_layout.addWidget(self.skip_chk)

            self.delete_chk = QtWidgets.QCheckBox("Delete rejected input images (DANGEROUS)")
            self.delete_chk.setChecked(False)
            left_layout.addWidget(self.delete_chk)

            self.variant_combo = QtWidgets.QComboBox()
            self.variant_combo.currentIndexChanged.connect(self._reload_current)
            self.variant_label = QtWidgets.QLabel("Variant")
            self.variant_yaml_btn = QtWidgets.QPushButton("Load YAML")
            self.variant_yaml_btn.clicked.connect(self._load_variants_yaml)
            form.addRow(self.variant_label, self._hbox(self.variant_combo, self.variant_yaml_btn))

            self.num_candidates_spin = QtWidgets.QSpinBox()
            self.num_candidates_spin.setRange(1, 20)
            self.num_candidates_spin.setValue(6)
            self.num_candidates_spin.valueChanged.connect(self._reload_current)
            self.candidates_label = QtWidgets.QLabel("Candidates")
            form.addRow(self.candidates_label, self.num_candidates_spin)

            self.kpt_std_spin = QtWidgets.QDoubleSpinBox()
            self.kpt_std_spin.setRange(0.0, 50.0)
            self.kpt_std_spin.setDecimals(1)
            self.kpt_std_spin.setSingleStep(0.5)
            self.kpt_std_spin.setValue(2.0)
            self.kpt_std_spin.valueChanged.connect(self._reload_current)
            self.kpt_jitter_label = QtWidgets.QLabel("Kpt Jitter (px)")
            form.addRow(self.kpt_jitter_label, self.kpt_std_spin)

            self.bbox_std_spin = QtWidgets.QDoubleSpinBox()
            self.bbox_std_spin.setRange(0.0, 50.0)
            self.bbox_std_spin.setDecimals(1)
            self.bbox_std_spin.setSingleStep(0.5)
            self.bbox_std_spin.setValue(1.0)
            self.bbox_std_spin.valueChanged.connect(self._reload_current)
            self.bbox_jitter_label = QtWidgets.QLabel("BBox Jitter (px)")
            form.addRow(self.bbox_jitter_label, self.bbox_std_spin)

            self.seed_spin = QtWidgets.QSpinBox()
            self.seed_spin.setRange(0, 2**31 - 1)
            self.seed_spin.setValue(0)
            self.seed_spin.valueChanged.connect(self._reload_current)
            self.seed_label = QtWidgets.QLabel("Seed")
            form.addRow(self.seed_label, self.seed_spin)

            self.reroll_btn = QtWidgets.QPushButton("Re-roll")
            self.reroll_btn.clicked.connect(self._reroll)
            left_layout.addWidget(self.reroll_btn)

            self.cand_combo = QtWidgets.QComboBox()
            self.cand_combo.currentIndexChanged.connect(self._render)
            self.pick_candidate_label = QtWidgets.QLabel("Pick Candidate")
            form.addRow(self.pick_candidate_label, self.cand_combo)

            self.show_kpt_labels_chk = QtWidgets.QCheckBox("Show keypoint labels")
            self.show_kpt_labels_chk.setChecked(True)
            self.show_kpt_labels_chk.stateChanged.connect(self._render)
            left_layout.addWidget(self.show_kpt_labels_chk)

            # Device selector (default CPU). GPU is optional and depends on PyTorch/CUDA compatibility.
            self.device_combo = QtWidgets.QComboBox()
            self.device_combo.addItem("CPU", "cpu")
            self.device_combo.addItem("GPU0", "cuda:0")
            self.device_combo.setCurrentIndex(0)
            self.device_combo.currentIndexChanged.connect(self._request_restart_worker)
            self.device_label = QtWidgets.QLabel("Device")
            form.addRow(self.device_label, self.device_combo)

            btn_row = QtWidgets.QHBoxLayout()
            self.load_btn = QtWidgets.QPushButton("Load")
            self.load_btn.clicked.connect(self._load_session)
            self.prev_btn = QtWidgets.QPushButton("Prev")
            self.prev_btn.clicked.connect(self._prev)
            self.next_btn = QtWidgets.QPushButton("Next")
            self.next_btn.clicked.connect(self._next)
            btn_row.addWidget(self.load_btn)
            btn_row.addWidget(self.prev_btn)
            btn_row.addWidget(self.next_btn)
            left_layout.addLayout(btn_row)

            act_row = QtWidgets.QHBoxLayout()

            self.edit_btn = QtWidgets.QPushButton("Edit")
            self.edit_btn.clicked.connect(self._edit_current)
            act_row.addWidget(self.edit_btn)

            self.clear_btn = QtWidgets.QPushButton("Clear")
            self.clear_btn.clicked.connect(self._clear_marks)
            act_row.addWidget(self.clear_btn)

            self.accept_btn = QtWidgets.QPushButton("Accept / Save")
            self.accept_btn.clicked.connect(self._accept)
            self.reject_btn = QtWidgets.QPushButton("Reject")
            self.reject_btn.clicked.connect(self._reject)
            act_row.addWidget(self.accept_btn)
            act_row.addWidget(self.reject_btn)
            left_layout.addLayout(act_row)

            left_layout.addStretch(1)

            self.image_label = QtWidgets.QLabel("Load a session to start")
            self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.image_label.setMinimumSize(640, 480)
            self.image_label.setStyleSheet("QLabel { background: #111; color: #ddd; }")
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setWidget(self.image_label)

            self.shortcuts_label = QtWidgets.QLabel("")
            self.shortcuts_label.setWordWrap(True)
            self.shortcuts_label.setStyleSheet("QLabel { color: #666; }")
            left_layout.addWidget(self.shortcuts_label)

            splitter = QtWidgets.QSplitter()
            splitter.addWidget(left)
            splitter.addWidget(self.scroll)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            self.setCentralWidget(splitter)

            self.status = self.statusBar()

            from yolo_labeler import load_variants

            self._variants = load_variants(None)
            for v in self._variants:
                self.variant_combo.addItem(str(v.get("name", "variant")))

            self._update_enabled()
            self._apply_lang()

            self._install_shortcuts()

            # Log unhandled exceptions to file for debugging sudden exits.
            def _excepthook(exc_type, exc, tb):
                text = "".join(traceback.format_exception(exc_type, exc, tb))
                try:
                    Path("yolo_labeler_gui_crash.log").write_text(text, encoding="utf-8")
                except Exception:
                    pass
                sys.__excepthook__(exc_type, exc, tb)

            sys.excepthook = _excepthook

        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
            super().resizeEvent(event)
            if self._pix_orig is not None:
                self._set_pixmap_fit(self._pix_orig)

        def _set_pixmap_fit(self, pix: QtGui.QPixmap) -> None:
            # Fit to viewport, keep aspect ratio.
            vp = self.scroll.viewport().size()
            scaled = pix.scaled(vp, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)

        def _toggle_lang(self) -> None:
            self._lang = "zh" if self._lang == "en" else "en"
            self._apply_lang()

        def _apply_lang(self) -> None:
            zh = self._lang == "zh"
            self.setWindowTitle("YOLO 姿态标注工具" if zh else "YOLO Pose Labeler")
            self.lang_btn.setText("English" if zh else "中文")

            self.project_yaml_label.setText("工程YAML" if zh else "Project YAML")
            self.model_label.setText("模型(.pt)" if zh else "Model (.pt)")
            self.input_label.setText("输入文件夹" if zh else "Input Folder")
            self.output_label.setText("输出文件夹" if zh else "Output Folder")
            self.rejected_label.setText("拒绝文件夹" if zh else "Rejected Folder")
            self.conf_label.setText("置信度" if zh else "Conf")
            self.iou_label.setText("IoU" if zh else "IoU")
            self.skip_chk.setText("跳过已输出标签" if zh else "Skip existing output labels")
            self.delete_chk.setText("拒绝时删除输入图片(危险)" if zh else "Delete rejected input images (DANGEROUS)")
            self.variant_label.setText("方案" if zh else "Variant")
            self.variant_yaml_btn.setText("导入YAML" if zh else "Load YAML")
            self.candidates_label.setText("候选数量" if zh else "Candidates")
            self.kpt_jitter_label.setText("关键点抖动(px)" if zh else "Kpt Jitter (px)")
            self.bbox_jitter_label.setText("整体抖动(px)" if zh else "BBox Jitter (px)")
            self.seed_label.setText("随机种子" if zh else "Seed")
            self.reroll_btn.setText("重新随机" if zh else "Re-roll")
            self.pick_candidate_label.setText("选择候选" if zh else "Pick Candidate")

            self.show_kpt_labels_chk.setText("显示关键点文字" if zh else "Show keypoint labels")
            self.device_label.setText("设备" if zh else "Device")

            if zh:
                self.shortcuts_label.setText(
                    "快捷键: 下一张 Right/D/Space | 上一张 Left/A | 保存 S | 拒绝 R | 编辑 E | 清除 Del/Backspace | 切换方案 V | 切换候选 C"
                )
            else:
                self.shortcuts_label.setText(
                    "Shortcuts: Next Right/D/Space | Prev Left/A | Save S | Reject R | Edit E | Clear Del/Backspace | Variant V | Candidate C"
                )

            self.model_btn.setText("选择" if zh else "Browse")
            self.input_btn.setText("选择" if zh else "Browse")
            self.output_btn.setText("选择" if zh else "Browse")
            self.rejected_btn.setText("选择" if zh else "Browse")
            self.project_yaml_btn.setText("选择" if zh else "Browse")
            self.load_btn.setText("加载" if zh else "Load")
            self.prev_btn.setText("上一张" if zh else "Prev")
            self.next_btn.setText("下一张" if zh else "Next")
            self.accept_btn.setText("确认保存" if zh else "Accept / Save")
            self.reject_btn.setText("拒绝" if zh else "Reject")
            self.edit_btn.setText("编辑" if zh else "Edit")
            self.clear_btn.setText("清除标注" if zh else "Clear")

        def _load_variants_yaml(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YAML", "", "YAML (*.yaml *.yml)")
            if not path:
                return
            try:
                import yaml
            except Exception:
                self._toast("Missing dependency: pyyaml (pip install pyyaml)")
                return
            try:
                data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
                if isinstance(data, dict) and "variants" in data:
                    data = data["variants"]
                if not isinstance(data, list):
                    raise ValueError("YAML must be a list or {variants: [...]}")
                variants: List[Dict[str, Any]] = []
                for v in data:
                    if not isinstance(v, dict) or "name" not in v:
                        raise ValueError("each variant must be a dict with 'name'")
                    variants.append(v)

                self._variants = variants
                self.variant_combo.blockSignals(True)
                self.variant_combo.clear()
                for v in self._variants:
                    self.variant_combo.addItem(str(v.get("name", "variant")))
                self.variant_combo.setCurrentIndex(0)
                self.variant_combo.blockSignals(False)
                self._reload_current()
            except Exception as e:
                self._toast(f"YAML error: {e}")

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
            try:
                self._stop_worker()
            finally:
                super().closeEvent(event)

        def _hbox(self, a: QtWidgets.QWidget, b: QtWidgets.QWidget) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(a)
            lay.addWidget(b)
            return w

        def _pick_model(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", self.model_edit.text(), "Model (*.pt)")
            if path:
                self.model_edit.setText(path)
                # If session already loaded, restart worker with new model.
                self._model_path = Path(path)
                self._request_restart_worker()

        def _pick_input(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_edit.text())
            if path:
                self.input_edit.setText(path)

        def _pick_output(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_edit.text())
            if path:
                self.output_edit.setText(path)
                if not self.rejected_edit.text().strip():
                    self.rejected_edit.setText(str(Path(path) / "rejected"))

        def _pick_rejected(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Rejected Folder", self.rejected_edit.text())
            if path:
                self.rejected_edit.setText(path)

        def _pick_project_yaml(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Project YAML",
                self.project_yaml_edit.text(),
                "YAML (*.yaml *.yml)",
            )
            if not path:
                return
            self.project_yaml_edit.setText(path)
            try:
                self._project = load_project_yaml(Path(path))
                self._class_names = dict(self._project.get("names", {}))
                self._kpt_names = self._project.get("kpt_names", None)
                self._skeleton = self._project.get("skeleton", None)
                self._expected_kpts = self._project.get("expected_kpts", None)
                self._allow_classes = self._project.get("allow_classes", None)

                defaults = self._project.get("defaults", {})
                if isinstance(defaults, dict):
                    if "conf" in defaults:
                        self.conf_spin.setValue(float(defaults["conf"]))
                    if "iou" in defaults:
                        self.iou_spin.setValue(float(defaults["iou"]))
                    if "candidates" in defaults:
                        self.num_candidates_spin.setValue(int(defaults["candidates"]))
                    if "kpt_jitter_px" in defaults:
                        self.kpt_std_spin.setValue(float(defaults["kpt_jitter_px"]))
                    if "bbox_jitter_px" in defaults:
                        self.bbox_std_spin.setValue(float(defaults["bbox_jitter_px"]))

                self._toast("Project YAML loaded")
                self._reload_current()
            except Exception as e:
                self._toast(f"Project YAML error: {e}")

        def _toast(self, msg: str) -> None:
            self.status.showMessage(msg, 5000)

        def _variant(self) -> Dict[str, Any]:
            idx = max(0, int(self.variant_combo.currentIndex()))
            return self._variants[idx]

        def _config(self) -> SessionConfig:
            assert self._model_path and self._input_dir and self._output_dir and self._rejected_dir
            return SessionConfig(
                model_path=self._model_path,
                input_dir=self._input_dir,
                output_dir=self._output_dir,
                rejected_dir=self._rejected_dir,
                conf=float(self.conf_spin.value()),
                iou=float(self.iou_spin.value()),
                skip_existing=bool(self.skip_chk.isChecked()),
                delete_rejected=bool(self.delete_chk.isChecked()),
            )

        def _out_label_path_for(self, image_path: Path) -> Path:
            cfg = self._config()
            return cfg.output_dir / "labels" / f"{image_path.stem}.txt"

        def _load_session(self) -> None:
            self._model_path = Path(self.model_edit.text().strip()) if self.model_edit.text().strip() else None
            self._input_dir = Path(self.input_edit.text().strip()) if self.input_edit.text().strip() else None
            self._output_dir = Path(self.output_edit.text().strip()) if self.output_edit.text().strip() else None
            self._rejected_dir = Path(self.rejected_edit.text().strip()) if self.rejected_edit.text().strip() else None

            if not self._model_path or not self._model_path.exists():
                self._toast("Invalid model path")
                return
            if not self._input_dir or not self._input_dir.exists():
                self._toast("Invalid input folder")
                return
            if not self._output_dir:
                self._toast("Invalid output folder")
                return
            if not self._rejected_dir:
                self._rejected_dir = self._output_dir / "rejected"
                self.rejected_edit.setText(str(self._rejected_dir))

            self._images = list_images(self._input_dir)
            if not self._images:
                self._toast("No images found")
                return

            self._idx = 0
            self._skip_to_next_unlabeled(+1)
            self._restart_worker()
            self._reload_current()

        def _request_restart_worker(self) -> None:
            # Avoid restarting while busy; defer until current inference finishes.
            self._restart_requested = True
            if not self._busy:
                self._restart_worker()

        def _current_device(self) -> str:
            # Always return a string like "cpu" or "cuda:0".
            if hasattr(self, "device_combo"):
                return str(self.device_combo.currentData())
            return "cpu"

        def _worker_program(self) -> tuple[str, list[str]]:
            py = sys.executable
            script = str((Path(__file__).parent / "yolo_infer_worker.py").resolve())
            return py, [script]

        def _stop_worker(self) -> None:
            if self._worker.state() == QtCore.QProcess.ProcessState.NotRunning:
                return
            try:
                self._worker.write((json.dumps({"type": "quit"}) + "\n").encode("utf-8"))
                self._worker.waitForBytesWritten(500)
            except Exception:
                pass
            self._worker.terminate()
            if not self._worker.waitForFinished(2000):
                self._worker.kill()
                self._worker.waitForFinished(2000)

        def _restart_worker(self) -> None:
            # Start/restart a separate process for inference.
            self._restart_requested = False

            if self._worker.state() != QtCore.QProcess.ProcessState.NotRunning:
                # If running, stop it first and wait for finished signal.
                self._stop_worker()
                if self._worker.state() != QtCore.QProcess.ProcessState.NotRunning:
                    # Still running; don't attempt to setProgram/start.
                    return

            self._stop_worker()
            self._worker_buf_out = ""
            self._worker_buf_err = ""
            self._worker_ready = False
            self._inflight_id = None
            self._inflight_path = None

            if not self._model_path:
                return

            py, args = self._worker_program()
            self._worker.setProgram(py)
            self._worker.setArguments(args)
            self._worker.start()
            if not self._worker.waitForStarted(5000):
                self.image_label.setText("Failed to start worker process")
                return

            init_msg = {"type": "init", "model": str(self._model_path), "device": self._current_device()}
            self._worker.write((json.dumps(init_msg, ensure_ascii=False) + "\n").encode("utf-8"))
            self._worker.waitForBytesWritten(2000)

        def _on_worker_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
            self._worker_ready = False
            self._inflight_id = None
            self._inflight_path = None
            if self._busy:
                self._busy = False
                self._update_enabled()
            self.image_label.setText(
                f"Worker exited (code={exit_code}, status={exit_status}). Switch device or restart."
            )

            # If a restart was requested (device/model change), restart now.
            if self._restart_requested and self._model_path:
                self._restart_worker()

        def _on_worker_stderr(self) -> None:
            data = bytes(self._worker.readAllStandardError()).decode("utf-8", errors="replace")
            if data:
                self._worker_buf_err += data

        def _on_worker_stdout(self) -> None:
            data = bytes(self._worker.readAllStandardOutput()).decode("utf-8", errors="replace")
            if not data:
                return
            self._worker_buf_out += data
            while "\n" in self._worker_buf_out:
                line, self._worker_buf_out = self._worker_buf_out.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue

                if msg.get("type") == "ready":
                    if msg.get("ok", True) is False:
                        self.image_label.setText(str(msg.get("error", "worker init failed")))
                        self._worker_ready = False
                    else:
                        self._worker_ready = True
                        if self._pending_reload:
                            self._pending_reload = False
                            self._reload_current()
                    continue

                if msg.get("type") == "result":
                    rid = msg.get("id")
                    if rid is None or self._inflight_id is None or int(rid) != int(self._inflight_id):
                        continue
                    self._inflight_id = None
                    self._inflight_path = None
                    self._busy = False
                    self._update_enabled()

                    if not bool(msg.get("ok", False)):
                        self.image_label.setText(str(msg.get("error", "")))
                        return

                    payload = msg.get("payload")
                    if isinstance(payload, dict):
                        self._apply_payload(payload)
                    if self._pending_reload:
                        self._pending_reload = False
                        self._reload_current()

                    # Handle deferred worker restart (e.g., device switch).
                    if self._restart_requested:
                        self._restart_worker()

        def _apply_payload(self, payload: Dict[str, Any]) -> None:
            import cv2
            from yolo_labeler import Box, Keypoint, PoseInstance

            image_path = payload.get("image")
            if not isinstance(image_path, str):
                return
            if self._images:
                current_path = str(self._images[self._idx])
                if image_path != current_path:
                    return

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                self.image_label.setText(f"Failed to read image: {image_path}")
                return

            candidates_raw = payload.get("candidates", [])
            candidates: List[List[Any]] = []
            for cand in candidates_raw:
                poses: List[Any] = []
                for pd in cand:
                    b = pd.get("bbox", [0, 0, 0, 0])
                    bbox = Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]), cls=int(pd.get("cls", 0)), conf=pd.get("conf", None))
                    kpts = []
                    for k in pd.get("kpts", []):
                        kpts.append(Keypoint(float(k[0]), float(k[1]), v=int(k[2]), conf=(None if k[3] is None else float(k[3]))))
                    poses.append(PoseInstance(bbox=bbox, kpts=kpts))
                candidates.append(poses)

            self._img_bgr = img_bgr
            self._candidates = candidates
            self.cand_combo.blockSignals(True)
            self.cand_combo.clear()
            for i in range(len(candidates)):
                self.cand_combo.addItem(f"{i}")
            self.cand_combo.setCurrentIndex(0)
            self.cand_combo.blockSignals(False)
            self._render()

            p = Path(image_path)
            self.status.showMessage(f"{self._idx + 1}/{len(self._images)}  {p.name}")

            if self._edit_mode:
                self._send_editor_update()

        def _skip_to_next_unlabeled(self, direction: int) -> None:
            if not self._images:
                return
            if not self.skip_chk.isChecked():
                return
            n = len(self._images)
            for _ in range(n):
                p = self._images[self._idx]
                if p.exists() and not self._out_label_path_for(p).exists():
                    return
                self._idx = (self._idx + direction) % n

        def _reroll(self) -> None:
            self.seed_spin.setValue(int(self.seed_spin.value()) + 1)

        def _reload_current(self) -> None:
            if not self._images or not self._model_path:
                return
            if self._busy:
                self._pending_reload = True
                return
            img_path = self._images[self._idx]
            if not img_path.exists():
                self._next()
                return

            if not self._worker_ready or self._worker.state() == QtCore.QProcess.ProcessState.NotRunning:
                self._pending_reload = True
                self._restart_worker()
                return

            self._busy = True
            self._update_enabled()
            self.status.showMessage(f"Loading {img_path.name} ...")
            self.image_label.setText("Running inference...")

            req_id = self._next_req_id
            self._next_req_id += 1
            self._inflight_id = int(req_id)
            self._inflight_path = str(img_path)

            msg = {
                "type": "infer",
                "id": int(req_id),
                "image": str(img_path),
                "conf": float(self.conf_spin.value()),
                "iou": float(self.iou_spin.value()),
                "variant_json": json.dumps({**self._variant(), "allow_classes": self._allow_classes}, ensure_ascii=False),
                "seed": int(self.seed_spin.value()),
                "num_candidates": int(self.num_candidates_spin.value()),
                "kpt_std_px": float(self.kpt_std_spin.value()),
                "bbox_std_px": float(self.bbox_std_spin.value()),
            }
            self._worker.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            self._worker.waitForBytesWritten(2000)

        def _render(self) -> None:
            if self._img_bgr is None or not self._candidates:
                return
            import cv2

            from yolo_labeler import draw_overlay

            img_path = self._images[self._idx]
            cand_idx = max(0, int(self.cand_combo.currentIndex()))
            poses = self._candidates[cand_idx]
            title = [
                f"{self._idx + 1}/{len(self._images)}  {img_path.name}",
                f"variant={self._variant().get('name','?')}  candidate={cand_idx}/{len(self._candidates) - 1}",
            ]

            vis_bgr = draw_overlay(
                self._img_bgr,
                poses,
                self._class_names,
                False,
                title,
                0,
                True,
                kpt_names=self._kpt_names,
                skeleton=[(int(a), int(b)) for a, b in self._skeleton] if self._skeleton else None,
                show_kpt_labels=bool(self.show_kpt_labels_chk.isChecked()),
            )
            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            h, w = vis_rgb.shape[:2]
            qimg = QtGui.QImage(vis_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg.copy())
            self._pix_orig = pix
            self._set_pixmap_fit(pix)

        def _accept(self) -> None:
            if self._img_bgr is None or not self._candidates or not self._images:
                return
            from yolo_labeler import save_sample

            cfg = self._config()
            img_path = self._images[self._idx]
            out_images = cfg.output_dir / "images"
            out_labels = cfg.output_dir / "labels"
            cand_idx = max(0, int(self.cand_combo.currentIndex()))
            poses = self._candidates[cand_idx]

            if self._expected_kpts is not None:
                for p in poses:
                    if len(getattr(p, "kpts", [])) != int(self._expected_kpts):
                        self._toast(f"Keypoint count mismatch: expected {self._expected_kpts}")
                        return
            save_sample(img_path, self._img_bgr, poses, out_images, out_labels)
            self._next()

        def _edit_current(self) -> None:
            # Start/keep one OpenCV editor window open.
            if self._img_bgr is None or not self._candidates:
                return
            if self.cand_combo.count() <= 0:
                return
            from yolo_labeler import PersistentPoseEditor

            if self._editor is None:
                self._editor = PersistentPoseEditor()
                self._editor.start()
            self._edit_mode = True
            if not self._editor_timer.isActive():
                self._editor_timer.start()
            self._send_editor_update()

        def _send_editor_update(self) -> None:
            if not self._edit_mode or self._editor is None:
                return
            if self._img_bgr is None or self._candidates is None:
                return
            if self.cand_combo.count() <= 0:
                return
            cand_idx = max(0, int(self.cand_combo.currentIndex()))
            if not (0 <= cand_idx < len(self._candidates)):
                return
            poses = self._candidates[cand_idx]
            from yolo_labeler import EditorUpdate

            self._editor.send_update(
                EditorUpdate(
                    image_bgr=self._img_bgr.copy(),
                    poses=poses,
                    class_names=self._class_names,
                    kpt_names=self._kpt_names,
                    skeleton=[(int(a), int(b)) for a, b in self._skeleton] if self._skeleton else None,
                    lang=self._lang,
                )
            )

        def _poll_editor(self) -> None:
            if self._editor is None:
                return
            try:
                while True:
                    res = self._editor.result_queue.get_nowait()
                    if res.closed:
                        self._edit_mode = False
                        self._editor = None
                        self._editor_timer.stop()
                        return
                    if res.ok:
                        if self._candidates is None or self.cand_combo.count() <= 0:
                            continue
                        cand_idx = max(0, int(self.cand_combo.currentIndex()))
                        if 0 <= cand_idx < len(self._candidates):
                            self._candidates[cand_idx] = res.poses
                            self._render()
            except Exception:
                return

        def _reject(self) -> None:
            if not self._images:
                return
            from yolo_labeler import safe_reject

            cfg = self._config()
            img_path = self._images[self._idx]
            safe_reject(img_path, cfg.rejected_dir, hard_delete=cfg.delete_rejected)
            self._next()

        def _prev(self) -> None:
            if not self._images:
                return
            self._idx = (self._idx - 1) % len(self._images)
            self._skip_to_next_unlabeled(-1)
            self._reload_current()

        def _next(self) -> None:
            if not self._images:
                return
            self._idx = (self._idx + 1) % len(self._images)
            self._skip_to_next_unlabeled(+1)
            self._reload_current()

        def _update_enabled(self) -> None:
            enabled = not self._busy
            for w in [
                self.load_btn,
                self.prev_btn,
                self.next_btn,
                self.accept_btn,
                self.reject_btn,
                self.model_btn,
                self.input_btn,
                self.output_btn,
                self.rejected_btn,
                self.variant_combo,
                self.cand_combo,
                self.num_candidates_spin,
                self.kpt_std_spin,
                self.bbox_std_spin,
                self.seed_spin,
                self.reroll_btn,
                self.edit_btn,
                self.clear_btn,
                self.show_kpt_labels_chk,
                self.device_combo,
            ]:
                w.setEnabled(enabled)

        def _install_shortcuts(self) -> None:
            # Common shortcuts (application-wide)
            def add(keys: str, fn) -> None:
                sc = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
                sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
                sc.activated.connect(fn)

            # Navigation
            add("Right", self._next)
            add("D", self._next)
            add("Space", self._next)
            add("Left", self._prev)
            add("A", self._prev)

            # Actions
            add("UP", self._accept)
            add("S", self._accept)
            add("DOWN", self._reject)
            add("R", self._reject)
            add("E", self._edit_current)
            add("Delete", self._clear_marks)
            add("Backspace", self._clear_marks)

            # Variants / candidates
            add("V", self._cycle_variant)
            add("C", self._cycle_candidate)

        def _cycle_variant(self) -> None:
            if self.variant_combo.count() <= 0:
                return
            i = (int(self.variant_combo.currentIndex()) + 1) % int(self.variant_combo.count())
            self.variant_combo.setCurrentIndex(i)

        def _cycle_candidate(self) -> None:
            if self.cand_combo.count() <= 0:
                return
            i = (int(self.cand_combo.currentIndex()) + 1) % int(self.cand_combo.count())
            self.cand_combo.setCurrentIndex(i)

        def _clear_marks(self) -> None:
            # Clear all instances for current candidate.
            if self._candidates is None:
                return
            if self.cand_combo.count() <= 0:
                return
            idx = max(0, int(self.cand_combo.currentIndex()))
            if 0 <= idx < len(self._candidates):
                self._candidates[idx] = []
                self._render()


def main() -> int:
    if not _HAS_QT:
        print("Missing dependency: PySide6. Install: pip install PySide6")
        return 2
    app = QtWidgets.QApplication(sys.argv)  # type: ignore[name-defined]
    w = MainWindow()  # type: ignore[name-defined]
    w.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
