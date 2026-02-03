import os
import json
import sys
import traceback
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2

from yolo_labeler import Box, Keypoint, PoseInstance, load_project_yaml, load_variants, save_sample, copy_pose_instances, clamp_kpt, clamp_box, point_in_box, nearest_kpt, draw_overlay, LineSeg

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

def list_images(input_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths: List[Path] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)

class InteractiveView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.Signal(float)
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        
        self._zoom_level = 1.0
        self._space_pressed = False
        self._line_mode = False
        self._create_mode = False
        self._line_start = None
        self._create_start = None
        self._temp_line = None
        self._temp_rect = None
        self._line_color = QtGui.QColor(255, 0, 0)
        self._rect_color = QtGui.QColor(0, 255, 0)

    def set_line_mode(self, enabled: bool):
        self._line_mode = enabled
        self._create_mode = False
        self._line_start = None
        if self._temp_line and self.scene():
            self.scene().removeItem(self._temp_line)
        self._temp_line = None

    def set_create_mode(self, enabled: bool):
        self._create_mode = enabled
        self._line_mode = False
        self._create_start = None
        if self._temp_rect and self.scene():
            self.scene().removeItem(self._temp_rect)
        self._temp_rect = None

    def set_line_color(self, color: QtGui.QColor):
        self._line_color = color

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = self.mapToScene(event.pos())
        if self._create_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._create_start = pos
            self._temp_rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(pos, pos))
            self._temp_rect.setPen(QtGui.QPen(self._rect_color, 2))
            self.scene().addItem(self._temp_rect)
            return

        if self._line_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            items = self.scene().items(pos)
            for item in items:
                if isinstance(item, KeypointItem):
                    pos = item.pos()
                    break
            
            if self._line_start is None:
                self._line_start = pos
                self._temp_line = QtWidgets.QGraphicsLineItem(QtCore.QLineF(pos, pos))
                self._temp_line.setPen(QtGui.QPen(self._line_color, 2))
                self.scene().addItem(self._temp_line)
            else:
                line = GuideLineItem(self._line_start, pos, self._line_color)
                self.scene().addItem(line)
                if self._temp_line:
                    self.scene().removeItem(self._temp_line)
                self._temp_line = None
                self._line_start = None
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = self.mapToScene(event.pos())
        if self._create_mode and self._create_start and self._temp_rect:
            rect = QtCore.QRectF(self._create_start, pos).normalized()
            self._temp_rect.setRect(rect)
            return

        if self._line_mode and self._line_start and self._temp_line:
            self._temp_line.setLine(QtCore.QLineF(self._line_start, pos))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._create_mode and self._create_start and self._temp_rect:
            rect = self._temp_rect.rect()
            self.scene().removeItem(self._temp_rect)
            self._temp_rect = None
            self._create_start = None
            if rect.width() > 5 and rect.height() > 5:
                main_win = self.window()
                if hasattr(main_win, '_on_bbox_drawn'):
                    main_win._on_bbox_drawn(rect)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.1 if angle > 0 else 0.9
            self._zoom_level *= factor
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom_level)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._space_pressed = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._space_pressed = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        super().keyReleaseEvent(event)

class KeypointItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, x: float, y: float, id: int, parent_pose: 'PoseItem'):
        super().__init__(-5, -5, 10, 10)
        self.setPos(x, y)
        self.id = id
        self.parent_pose = parent_pose
        self.setBrush(QtGui.QColor(255, 255, 0))
        self.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 1))
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.v = 2
        
    def set_visible_state(self, v: int):
        self.v = v
        if v == 0:
            self.setOpacity(0.3)
            self.setBrush(QtGui.QColor(150, 150, 150))
        else:
            self.setOpacity(1.0)
            self.setBrush(QtGui.QColor(255, 255, 0))

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.parent_pose.update_geometry()
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            new_v = 0 if self.v > 0 else 2
            self.set_visible_state(new_v)
            self.parent_pose.update_geometry()
        super().mousePressEvent(event)

class GuideLineItem(QtWidgets.QGraphicsLineItem):
    def __init__(self, p1: QtCore.QPointF, p2: QtCore.QPointF, color: QtGui.QColor):
        super().__init__(QtCore.QLineF(p1, p2))
        self.setPen(QtGui.QPen(color, 2, QtCore.Qt.PenStyle.DashLine))
        self.setZValue(-1)

class PoseItem(QtWidgets.QGraphicsItemGroup):
    def __init__(self, pose: PoseInstance, class_names: Dict[int, str], skeleton: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | 
                      QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.pose = pose
        self.skeleton_indices = skeleton
        self.class_names = class_names
        self.rotation_angle = 0.0
        
        b = pose.bbox
        self.rect_item = QtWidgets.QGraphicsRectItem(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)
        self.rect_item.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255), 2))
        self.addToGroup(self.rect_item)
        
        name = class_names.get(b.cls, str(b.cls))
        self.label_item = QtWidgets.QGraphicsSimpleTextItem(name)
        self.label_item.setPos(b.x1, b.y1 - 20)
        self.label_item.setBrush(QtGui.QColor(0, 255, 255))
        font = self.label_item.font()
        font.setPixelSize(14)
        font.setBold(True)
        self.label_item.setFont(font)
        self.addToGroup(self.label_item)
        
        self.kpt_items: List[KeypointItem] = []
        for i, kp in enumerate(pose.kpts):
            ki = KeypointItem(kp.x, kp.y, i, self)
            ki.set_visible_state(int(kp.v))
            self.kpt_items.append(ki)
        
        self.skeleton_lines: List[QtWidgets.QGraphicsLineItem] = []
        
    def add_to_scene(self, scene: QtWidgets.QGraphicsScene):
        scene.addItem(self)
        for ki in self.kpt_items:
            scene.addItem(ki)
        self.update_geometry()
        
    def update_geometry(self):
        for ln in self.skeleton_lines:
            if ln.scene():
                ln.scene().removeItem(ln)
        self.skeleton_lines.clear()
        
        visible_pts = {ki.id: ki.pos() for ki in self.kpt_items if ki.v > 0}
        
        if self.skeleton_indices:
            for a_i, b_i in self.skeleton_indices:
                if a_i in visible_pts and b_i in visible_pts:
                    ln = QtWidgets.QGraphicsLineItem(QtCore.QLineF(visible_pts[a_i], visible_pts[b_i]))
                    ln.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
                    self.skeleton_lines.append(ln)
                    self.scene().addItem(ln)
        else:
            ids = sorted(visible_pts.keys())
            for i in range(len(ids)-1):
                ln = QtWidgets.QGraphicsLineItem(QtCore.QLineF(visible_pts[ids[i]], visible_pts[ids[i+1]]))
                ln.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
                self.skeleton_lines.append(ln)
                self.scene().addItem(ln)

        if visible_pts:
            xs = [p.x() for p in visible_pts.values()]
            ys = [p.y() for p in visible_pts.values()]
            pad = 10
            new_rect = QtCore.QRectF(min(xs)-pad, min(ys)-pad, max(xs)-min(xs)+2*pad, max(ys)-min(ys)+2*pad)
            self.rect_item.setRect(new_rect)
            self.label_item.setPos(new_rect.topLeft() + QtCore.QPointF(0, -20))

    def rotate_pose(self, angle_delta: float):
        self.rotation_angle += angle_delta
        center = self.rect_item.rect().center()
        t = QtGui.QTransform().translate(center.x(), center.y()).rotate(self.rotation_angle).translate(-center.x(), -center.y())
        self.setTransform(t)

    def get_pose(self) -> PoseInstance:
        kpts = []
        for ki in self.kpt_items:
            p = ki.pos()
            kpts.append(Keypoint(x=p.x(), y=p.y(), v=ki.v, conf=None))
        r = self.rect_item.rect()
        br = Box(r.left(), r.top(), r.right(), r.bottom(), cls=self.pose.bbox.cls, conf=self.pose.bbox.conf)
        return PoseInstance(bbox=br, kpts=kpts)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._lang = "en"
        self.setWindowTitle("YOLO Pose Integrated Labeler")
        self.resize(1600, 1000)
        
        self._model_path = None
        self._input_dir = None
        self._output_dir = None
        self._rejected_dir = None
        self._images = []
        self._idx = 0
        self._candidates = []
        self._busy = False
        self._class_names = {}
        self._skeleton = None
        self._expected_kpts = 5
        
        self._delete_confirm_item = None
        self._delete_confirm_time = 0
        
        self._init_ui()
        self._init_worker()
        self._install_shortcuts()
        self._apply_lang()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(320)
        self.sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        layout.addWidget(sidebar)
        
        form = QtWidgets.QFormLayout()
        self.sidebar_layout.addLayout(form)
        
        self.lang_btn = QtWidgets.QPushButton("中文")
        self.lang_btn.clicked.connect(self._toggle_lang)
        self.sidebar_layout.addWidget(self.lang_btn)
        
        self.project_yaml_edit = QtWidgets.QLineEdit()
        self.project_yaml_btn = QtWidgets.QPushButton("...")
        self.project_yaml_btn.clicked.connect(self._pick_project_yaml)
        self.project_yaml_label = QtWidgets.QLabel("Project YAML")
        form.addRow(self.project_yaml_label, self._hbox(self.project_yaml_edit, self.project_yaml_btn))
        
        self.model_edit = QtWidgets.QLineEdit()
        self.model_btn = QtWidgets.QPushButton("...")
        self.model_btn.clicked.connect(self._pick_model)
        self.model_label = QtWidgets.QLabel("Model (.pt)")
        form.addRow(self.model_label, self._hbox(self.model_edit, self.model_btn))
        
        self.input_edit = QtWidgets.QLineEdit()
        self.input_btn = QtWidgets.QPushButton("...")
        self.input_btn.clicked.connect(self._pick_input)
        self.input_label = QtWidgets.QLabel("Input Folder")
        form.addRow(self.input_label, self._hbox(self.input_edit, self.input_btn))
        
        self.output_edit = QtWidgets.QLineEdit()
        self.output_btn = QtWidgets.QPushButton("...")
        self.output_btn.clicked.connect(self._pick_output)
        self.output_label = QtWidgets.QLabel("Output Folder")
        form.addRow(self.output_label, self._hbox(self.output_edit, self.output_btn))
        
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_label = QtWidgets.QLabel("Conf")
        form.addRow(self.conf_label, self.conf_spin)
        
        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setValue(0.70)
        self.iou_label = QtWidgets.QLabel("IoU")
        form.addRow(self.iou_label, self.iou_spin)
        
        self.variant_combo = QtWidgets.QComboBox()
        self.variant_combo.currentIndexChanged.connect(self._reload_current)
        self.variant_label = QtWidgets.QLabel("Variant")
        form.addRow(self.variant_label, self.variant_combo)
        
        self.cand_combo = QtWidgets.QComboBox()
        self.cand_combo.currentIndexChanged.connect(self._display_current_candidate)
        self.pick_candidate_label = QtWidgets.QLabel("Candidate")
        form.addRow(self.pick_candidate_label, self.cand_combo)

        self.create_mode_chk = QtWidgets.QCheckBox("Create Pose Mode (N)")
        self.create_mode_chk.stateChanged.connect(lambda s: self.view.set_create_mode(s == QtCore.Qt.CheckState.Checked.value))
        self.sidebar_layout.addWidget(self.create_mode_chk)

        self.line_mode_chk = QtWidgets.QCheckBox("Draw Line Mode (L)")
        self.line_mode_chk.stateChanged.connect(lambda s: self.view.set_line_mode(s == QtCore.Qt.CheckState.Checked.value))
        self.sidebar_layout.addWidget(self.line_mode_chk)

        self.color_btn = QtWidgets.QPushButton("Pick Line Color")
        self.color_btn.clicked.connect(self._pick_line_color)
        self.sidebar_layout.addWidget(self.color_btn)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Session")
        self.load_btn.clicked.connect(self._load_session)
        btn_row.addWidget(self.load_btn)
        self.sidebar_layout.addLayout(btn_row)
        
        nav_row = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Prev (A)")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn = QtWidgets.QPushButton("Next (D)")
        self.next_btn.clicked.connect(self._next)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        self.sidebar_layout.addLayout(nav_row)
        
        act_row = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton("Save (S)")
        self.accept_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.accept_btn.clicked.connect(self._accept)
        self.reject_btn = QtWidgets.QPushButton("Reject (R)")
        self.reject_btn.clicked.connect(self._reject)
        act_row.addWidget(self.accept_btn)
        act_row.addWidget(self.reject_btn)
        self.sidebar_layout.addLayout(act_row)
        
        self.clear_btn = QtWidgets.QPushButton("Clear Marks (Del)")
        self.clear_btn.clicked.connect(self._clear_all_marks)
        self.sidebar_layout.addWidget(self.clear_btn)
        
        self.sidebar_layout.addStretch(1)
        self.shortcuts_label = QtWidgets.QLabel()
        self.shortcuts_label.setWordWrap(True)
        self.shortcuts_label.setStyleSheet("color: #aaa; background: #333; padding: 5px;")
        self.sidebar_layout.addWidget(self.shortcuts_label)
        
        self.scene = QtWidgets.QGraphicsScene()
        self.view = InteractiveView()
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        layout.addWidget(self.view)
        
        self.status = self.statusBar()

    def _on_bbox_drawn(self, rect: QtCore.QRectF):
        if not self._images or self.cand_combo.currentIndex() < 0:
            return
        k = self._expected_kpts if self._expected_kpts is not None else 5
        kpts = []
        cx, cy = rect.center().x(), rect.center().y()
        rw, rh = rect.width(), rect.height()
        if k == 5:
            pts = [(cx, cy), (rect.left(), rect.top()), (rect.right(), rect.top()), 
                   (rect.left(), rect.bottom()), (rect.right(), rect.bottom())]
            for px, py in pts:
                kpts.append(Keypoint(float(px), float(py), v=2, conf=None))
        else:
            cols = int(np.ceil(np.sqrt(k)))
            rows = int(np.ceil(k / cols))
            for i in range(k):
                r = i // cols
                c = i % cols
                px = rect.left() + (c + 0.5) * (rw / cols)
                py = rect.top() + (r + 0.5) * (rh / rows)
                kpts.append(Keypoint(float(px), float(py), v=2, conf=None))
        new_bbox = Box(rect.left(), rect.top(), rect.right(), rect.bottom(), cls=0, conf=1.0)
        new_pose = PoseInstance(bbox=new_bbox, kpts=kpts)
        if self._candidates is not None:
            self._candidates[self.cand_combo.currentIndex()].append(new_pose)
        item = PoseItem(new_pose, self._class_names, self._skeleton)
        item.add_to_scene(self.scene)
        item.setSelected(True)
        self._toast("New pose created")

    def _init_worker(self):
        self._worker = QtCore.QProcess(self)
        self._worker.readyReadStandardOutput.connect(self._on_worker_stdout)
        self._worker_ready = False
        self._worker_buf = ""

    def _restart_worker(self):
        if self._worker.state() != QtCore.QProcess.ProcessState.NotRunning:
            self._worker.terminate()
            self._worker.waitForFinished(1000)
        py = sys.executable
        script = str(Path(__file__).parent / "yolo_infer_worker.py")
        self._worker.setProgram(py)
        self._worker.setArguments([script])
        self._worker.start()
        if self._worker.waitForStarted(5000):
            init = {"type": "init", "model": str(self._model_path), "device": "cpu"}
            self._worker.write((json.dumps(init) + "\n").encode("utf-8"))

    def _on_worker_stdout(self):
        raw_data = self._worker.readAllStandardOutput()
        data = bytes(raw_data.data()).decode("utf-8", errors="replace")
        self._worker_buf += data
        while "\n" in self._worker_buf:
            line, self._worker_buf = self._worker_buf.split("\n", 1)
            try:
                msg = json.loads(line)
                if msg.get("type") == "ready":
                    self._worker_ready = True
                    self._reload_current()
                elif msg.get("type") == "result":
                    self._busy = False
                    if msg.get("ok"):
                        self._handle_result(msg["payload"])
                    else:
                        self._toast(msg.get("error"))
            except: pass

    def _handle_result(self, payload):
        cands_raw = payload["candidates"]
        self._candidates = []
        for cand in cands_raw:
            poses = []
            for p in cand:
                b = p["bbox"]
                bbox = Box(b[0], b[1], b[2], b[3], cls=p["cls"], conf=p["conf"])
                kpts = [Keypoint(k[0], k[1], v=k[2], conf=k[3]) for k in p["kpts"]]
                poses.append(PoseInstance(bbox=bbox, kpts=kpts))
            self._candidates.append(poses)
        self.cand_combo.blockSignals(True)
        self.cand_combo.clear()
        for i in range(len(self._candidates)):
            self.cand_combo.addItem(f"Variant {i}")
        self.cand_combo.setCurrentIndex(0)
        self.cand_combo.blockSignals(False)
        self._display_current_candidate()

    def _display_current_candidate(self):
        if not self._candidates or self.cand_combo.currentIndex() < 0: return
        self.scene.clear()
        img_path = self._images[self._idx]
        pix = QtGui.QPixmap(str(img_path))
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(pix.rect())
        poses = self._candidates[self.cand_combo.currentIndex()]
        for p in poses:
            item = PoseItem(p, self._class_names, self._skeleton)
            item.add_to_scene(self.scene)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.status.showMessage(f"Image {self._idx + 1}/{len(self._images)}: {img_path.name}")

    def _load_session(self):
        if not self.model_edit.text() or not self.input_edit.text():
            self._toast("Please select model and input folder")
            return
        self._model_path = Path(self.model_edit.text())
        self._input_dir = Path(self.input_edit.text())
        self._output_dir = Path(self.output_edit.text()) if self.output_edit.text() else Path("output")
        self._images = list_images(self._input_dir)
        if not self._images:
            self._toast("No images found")
            return
        self._idx = 0
        self._restart_worker()

    def _reload_current(self):
        if not self._worker_ready or self._busy or not self._images: return
        self._busy = True
        img_path = self._images[self._idx]
        vars_list = load_variants(None)
        v = vars_list[self.variant_combo.currentIndex()] if self.variant_combo.count() > 0 else vars_list[0]
        msg = {"type": "infer", "id": self._idx, "image": str(img_path), "conf": self.conf_spin.value(), "iou": self.iou_spin.value(), "variant_json": json.dumps(v)}
        self._worker.write((json.dumps(msg) + "\n").encode("utf-8"))

    def _next(self):
        if self._images:
            self._idx = (self._idx + 1) % len(self._images)
            self._reload_current()

    def _prev(self):
        if self._images:
            self._idx = (self._idx - 1) % len(self._images)
            self._reload_current()

    def _accept(self):
        if not self._images or not self._output_dir: return
        current_poses = [item.get_pose() for item in self.scene.items() if isinstance(item, PoseItem)]
        img_path = self._images[self._idx]
        img_raw = cv2.imread(str(img_path))
        if img_raw is not None:
            save_sample(img_path, img_raw, current_poses, self._output_dir / "images", self._output_dir / "labels")
        self._next()

    def _reject(self):
        self._next()

    def _clear_all_marks(self):
        for item in self.scene.items():
            if isinstance(item, (PoseItem, KeypointItem, QtWidgets.QGraphicsLineItem)):
                self.scene.removeItem(item)

    def _pick_line_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.view.set_line_color(color)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Delete or event.key() == QtCore.Qt.Key.Key_Backspace:
            selected = self.scene.selectedItems()
            if selected:
                item = selected[0]
                target = item if isinstance(item, PoseItem) else getattr(item, 'parent_pose', None)
                if target:
                    now = time.time()
                    if self._delete_confirm_item == target and (now - self._delete_confirm_time) < 2.0:
                        self.scene.removeItem(target)
                        for k in target.kpt_items: self.scene.removeItem(k)
                        for l in target.skeleton_lines: self.scene.removeItem(l)
                        self._delete_confirm_item = None
                    else:
                        self._delete_confirm_item = target
                        self._delete_confirm_time = now
                        self._toast("Press Delete again to confirm removal")
        elif event.key() == QtCore.Qt.Key.Key_BracketLeft:
            for item in self.scene.selectedItems():
                if isinstance(item, PoseItem): item.rotate_pose(-5)
        elif event.key() == QtCore.Qt.Key.Key_BracketRight:
            for item in self.scene.selectedItems():
                if isinstance(item, PoseItem): item.rotate_pose(5)
        elif event.key() == QtCore.Qt.Key.Key_L:
            self.line_mode_chk.toggle()
        elif event.key() == QtCore.Qt.Key.Key_N:
            self.create_mode_chk.toggle()
        super().keyPressEvent(event)

    def _toggle_lang(self):
        self._lang = "zh" if self._lang == "en" else "en"
        self._apply_lang()

    def _apply_lang(self):
        zh = self._lang == "zh"
        self.setWindowTitle("YOLO 姿态集成标注工具" if zh else "YOLO Pose Integrated Labeler")
        self.lang_btn.setText("English" if zh else "中文")
        self.project_yaml_label.setText("工程YAML" if zh else "Project YAML")
        self.model_label.setText("模型(.pt)" if zh else "Model (.pt)")
        self.input_label.setText("输入文件夹" if zh else "Input Folder")
        self.output_label.setText("输出文件夹" if zh else "Output Folder")
        self.load_btn.setText("启动会话" if zh else "Load Session")
        self.accept_btn.setText("保存 (S)" if zh else "Save (S)")
        self.reject_btn.setText("拒绝 (R)" if zh else "Reject (R)")
        self.clear_btn.setText("清除标注 (Del)" if zh else "Clear Marks (Del)")
        self.create_mode_chk.setText("绘框模式 (N)" if zh else "Create Pose Mode (N)")
        self.line_mode_chk.setText("画辅助线模式 (L)" if zh else "Draw Line Mode (L)")
        self.color_btn.setText("选取线条颜色" if zh else "Pick Line Color")
        self.shortcuts_label.setText(
            "快捷键:\nA/D: 上一张/下一张\nS/R: 保存/拒绝\nDel: 双击删除目标\nN: 绘框模式\nL: 画线模式\nCtrl+滚轮: 缩放\n空格+拖拽: 平移\n[/]: 旋转选中框" if zh else
            "Shortcuts:\nA/D: Prev/Next\nS/R: Save/Reject\nDel: Double-tap delete\nN: Create mode\nL: Line mode\nCtrl+Wheel: Zoom\nSpace+Drag: Pan\n[/]: Rotate selected"
        )

    def _pick_project_yaml(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YAML", "", "YAML (*.yaml *.yml)")
        if path:
            self.project_yaml_edit.setText(path)
            data = load_project_yaml(Path(path))
            self._class_names = data["names"]
            self._skeleton = data["skeleton"]
            self._expected_kpts = data.get("expected_kpts", 5)

    def _pick_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", "", "Model (*.pt)")
        if path: self.model_edit.setText(path)

    def _pick_input(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path: self.input_edit.setText(path)

    def _pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path: self.output_edit.setText(path)

    def _toast(self, msg):
        self.status.showMessage(str(msg), 3000)

    def _hbox(self, a, b):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0,0,0,0)
        l.addWidget(a)
        l.addWidget(b)
        return w

    def _install_shortcuts(self):
        def add(k, fn):
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.activated.connect(fn)
        add("D", self._next)
        add("Right", self._next)
        add("A", self._prev)
        add("Left", self._prev)
        add("S", self._accept)
        add("R", self._reject)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
