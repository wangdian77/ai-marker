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

from yolo_labeler import Box, Keypoint, PoseInstance, load_project_yaml, load_variants, save_sample, safe_reject, copy_pose_instances, clamp_kpt, clamp_box, point_in_box, nearest_kpt, draw_overlay, LineSeg

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
    num_candidates: int
    seed: int
    kpt_std_px: float
    bbox_std_px: float

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
        pos = self.mapToScene(event.position().toPoint())

        # Penetrating hit-test priority (design doc): kpt > guide/skeleton line > small bbox > large bbox.
        if (not self._create_mode) and (not self._line_mode) and event.button() == QtCore.Qt.MouseButton.LeftButton:
            try:
                items = self.scene().items(pos)
                # 2) Lines (guide or skeleton)
                for it in items:
                    if isinstance(it, (GuideLineItem, SkeletonLineItem)):
                        self.scene().clearSelection()
                        it.setSelected(True)
                        event.accept()
                        return
                # 3) Smallest PoseItem
                poses = [it for it in items if isinstance(it, PoseItem)]
                if poses:
                    target = min(
                        poses,
                        key=lambda p: float(p.boundingRect().width() * p.boundingRect().height()),
                    )
                    self.scene().clearSelection()
                    target.setSelected(True)
                    event.accept()
                    return
            except Exception:
                pass

        if self._create_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            # If an instance is selected and the user draws near it, treat as overwrite.
            main_win = self.window()
            if isinstance(main_win, MainWindow):
                sel = main_win.scene.selectedItems() if hasattr(main_win, "scene") else []
                if sel:
                    it0 = sel[0]
                    pose = it0 if isinstance(it0, PoseItem) else getattr(it0, "parent_pose", None)
                    if isinstance(pose, PoseItem):
                        try:
                            r = pose.rect_item.rect()
                            center = pose.mapToScene(r.center())
                            size = float(max(r.width(), r.height()))
                            dist = float(((pos.x() - center.x()) ** 2 + (pos.y() - center.y()) ** 2) ** 0.5)
                            if dist <= 2.0 * max(1.0, size):
                                main_win._overwrite_target = pose
                                # Ghost the target to signal overwrite intent.
                                pose.rect_item.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2, QtCore.Qt.PenStyle.DashLine))
                                pose.label_item.setOpacity(0.3)
                                for k in pose.kpt_items:
                                    k.setOpacity(0.3)
                                for l in getattr(pose, "skeleton_lines", []):
                                    l.setOpacity(0.3)
                            else:
                                pose.setSelected(False)
                                main_win._overwrite_target = None
                        except Exception:
                            main_win._overwrite_target = None

            self._create_start = pos
            self._temp_rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(pos, pos))
            self._temp_rect.setPen(QtGui.QPen(self._rect_color, 2))
            self.scene().addItem(self._temp_rect)
            return

        if self._line_mode and event.button() == QtCore.Qt.MouseButton.RightButton:
            if self._temp_line and self.scene():
                self.scene().removeItem(self._temp_line)
            self._temp_line = None
            self._line_start = None
            event.accept()
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
        pos = self.mapToScene(event.position().toPoint())
        if self._create_mode and self._create_start and self._temp_rect:
            rect = QtCore.QRectF(self._create_start, pos).normalized()
            try:
                self._temp_rect.setRect(rect)
            except RuntimeError:
                # Under rare conditions the scene may be cleared while dragging.
                # Drop the temp item to avoid crashing the UI.
                self._temp_rect = None
                self._create_start = None
            return

        if self._line_mode and self._line_start and self._temp_line:
            try:
                self._temp_line.setLine(QtCore.QLineF(self._line_start, pos))
            except RuntimeError:
                self._temp_line = None
                self._line_start = None
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
                if isinstance(main_win, MainWindow):
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
            # Default: use wheel to cycle candidates (like legacy GUI's `C`).
            main_win = self.window()
            if hasattr(main_win, "_cycle_candidate_by_wheel"):
                getattr(main_win, "_cycle_candidate_by_wheel")(int(event.angleDelta().y()))
                event.accept()
                return
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
        # Larger hit target; keep size constant on zoom for easier dragging.
        super().__init__(-10, -10, 20, 20)
        self.setPos(x, y)
        self.id = id
        self.parent_pose = parent_pose
        self.setBrush(QtGui.QColor(255, 255, 0))
        self.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 1))
        self.setZValue(20)
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.v = 2

    def hoverEnterEvent(self, event):
        try:
            pen = self.pen()
            pen.setWidth(2)
            self.setPen(pen)
        except Exception:
            pass
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        try:
            pen = self.pen()
            pen.setWidth(1)
            self.setPen(pen)
        except Exception:
            pass
        super().hoverLeaveEvent(event)

    def set_visible_state(self, v: int):
        self.v = v
        if v == 0:
            self.setOpacity(0.3)
            self.setBrush(QtGui.QColor(150, 150, 150))
            self.setZValue(5)
        else:
            self.setOpacity(1.0)
            self.setBrush(QtGui.QColor(255, 255, 0))
            self.setZValue(20)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self.scene():
                # Optional: snapping to guideline items.
                win = self.window()
                if hasattr(win, "_snap_kpt_to_guides"):
                    value = getattr(win, "_snap_kpt_to_guides")(value)

                # Optional: Alt-drag moves the whole instance.
                if QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    try:
                        old = self.pos()
                        new = QtCore.QPointF(value)
                        delta = new - old
                        if abs(delta.x()) > 1e-6 or abs(delta.y()) > 1e-6:
                            self.parent_pose._move_exclude = self
                            self.parent_pose.setPos(self.parent_pose.pos() + delta)
                            self.parent_pose._move_exclude = None
                    except Exception:
                        pass

                self.parent_pose.update_geometry()
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            win = self.window()
            if hasattr(win, "_push_undo"):
                getattr(win, "_push_undo")()
            p = self.parent_pose
            expected = 5
            win = self.window()
            if hasattr(win, '_expected_kpts'):
                attr = getattr(win, '_expected_kpts')
                if attr is not None:
                    expected = int(attr)
            
            # Count current visible points
            visible_count = len([k for k in p.kpt_items if k.v > 0])
            
            if visible_count > expected:
                # Physical remove if we have extras
                if self in p.kpt_items:
                    p.kpt_items.remove(self)
                if self.scene():
                    self.scene().removeItem(self)
            else:
                # Toggle hide if at or below minimum
                self.set_visible_state(0)
            
            p.update_geometry()
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            win = self.window()
            if hasattr(win, "_push_undo"):
                getattr(win, "_push_undo")()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

class GuideLineItem(QtWidgets.QGraphicsLineItem):
    def __init__(self, p1: QtCore.QPointF, p2: QtCore.QPointF, color: QtGui.QColor):
        super().__init__(QtCore.QLineF(p1, p2))
        self.setPen(QtGui.QPen(color, 2, QtCore.Qt.PenStyle.DashLine))
        self.setZValue(10)
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        
    def paint(self, painter, option, widget=None):
        if self.isSelected():
            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 3, QtCore.Qt.PenStyle.SolidLine))
        super().paint(painter, option, widget)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            win = self.window()
            if hasattr(win, "_push_undo"):
                getattr(win, "_push_undo")()
            # Immediate delete on right click
            if self.scene():
                self.scene().removeItem(self)
            event.accept()
            return
        super().mousePressEvent(event)


class SkeletonLineItem(QtWidgets.QGraphicsLineItem):
    def __init__(self, p1: QtCore.QPointF, p2: QtCore.QPointF, parent_pose: "PoseItem"):
        super().__init__(QtCore.QLineF(p1, p2))
        self.parent_pose = parent_pose
        self.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
        self.setZValue(10)
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            try:
                if self.scene():
                    self.scene().clearSelection()
                self.parent_pose.setSelected(True)
            except Exception:
                pass
            event.accept()
            return
        super().mousePressEvent(event)

class PoseItem(QtWidgets.QGraphicsObject):
    def __init__(self, pose: PoseInstance, class_names: Dict[int, str], skeleton: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        )
        self.setAcceptHoverEvents(True)
        self.pose = pose
        self.skeleton_indices = skeleton
        self.class_names = class_names
        self.rotation_angle = 0.0
        self._suspend_updates = False
        self._move_exclude: Optional[KeypointItem] = None
        
        b = pose.bbox
        # Ensure annotations render above the image pixmap (which uses z=-100).
        # Hit-testing priority is handled explicitly in InteractiveView; we don't rely on z for that.
        self.setZValue(1)
        self.rect_item = QtWidgets.QGraphicsRectItem(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1, self)
        self._base_pen = QtGui.QPen(QtGui.QColor(0, 255, 255), 2)
        self.rect_item.setPen(QtGui.QPen(self._base_pen))
        self.rect_item.setBrush(QtGui.QColor(0, 255, 255, 20))
        
        name = class_names.get(b.cls, str(b.cls))
        self.label_item = QtWidgets.QGraphicsSimpleTextItem(name, self)
        self.label_item.setPos(b.x1, b.y1 - 20)
        self.label_item.setBrush(QtGui.QColor(0, 255, 255))
        font = self.label_item.font()
        font.setPixelSize(14)
        font.setBold(True)
        self.label_item.setFont(font)
        
        self.kpt_items: List[KeypointItem] = []
        for i, kp in enumerate(pose.kpts):
            ki = KeypointItem(kp.x, kp.y, i, self)
            ki.set_visible_state(int(kp.v))
            ki.setZValue(20)
            self.kpt_items.append(ki)
        
        self.skeleton_lines: List[QtWidgets.QGraphicsLineItem] = []

    def itemChange(self, change, value):
        # Keep keypoints in sync when dragging the bbox/instance.
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            try:
                new_pos = QtCore.QPointF(value)
                delta = new_pos - self.pos()
                if abs(delta.x()) > 1e-6 or abs(delta.y()) > 1e-6:
                    self._suspend_updates = True
                    for ki in self.kpt_items:
                        if self._move_exclude is not None and ki is self._move_exclude:
                            continue
                        ki.setPos(ki.pos() + delta)
                    self._suspend_updates = False
                    self.update_geometry()
            except Exception:
                self._suspend_updates = False
        return super().itemChange(change, value)

    def boundingRect(self):
        return self.rect_item.boundingRect()

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 2, QtCore.Qt.PenStyle.DashLine))
            painter.drawRect(self.boundingRect())

    def hoverEnterEvent(self, event):
        try:
            pen = QtGui.QPen(self._base_pen)
            pen.setWidth(4)
            self.rect_item.setPen(pen)
        except Exception:
            pass
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        try:
            self.rect_item.setPen(QtGui.QPen(self._base_pen))
        except Exception:
            pass
        super().hoverLeaveEvent(event)
            
    def add_to_scene(self, scene: QtWidgets.QGraphicsScene):
        scene.addItem(self)
        for ki in self.kpt_items:
            scene.addItem(ki)
        self.update_geometry()
        
    def update_geometry(self):
        if getattr(self, "_suspend_updates", False):
            return
        for ln in self.skeleton_lines:
            if ln.scene():
                ln.scene().removeItem(ln)
        self.skeleton_lines.clear()
        
        visible_pts = {ki.id: ki.pos() for ki in self.kpt_items if ki.v > 0}
        
        if self.skeleton_indices:
            for a_i, b_i in self.skeleton_indices:
                if a_i in visible_pts and b_i in visible_pts:
                    ln = SkeletonLineItem(visible_pts[a_i], visible_pts[b_i], self)
                    self.skeleton_lines.append(ln)
                    self.scene().addItem(ln)
        else:
            ids = sorted(visible_pts.keys())
            for i in range(len(ids)-1):
                ln = SkeletonLineItem(visible_pts[ids[i]], visible_pts[ids[i + 1]], self)
                self.skeleton_lines.append(ln)
                self.scene().addItem(ln)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            win = self.window()
            if hasattr(win, "_push_undo"):
                getattr(win, "_push_undo")()
        super().mousePressEvent(event)

    def set_class(self, cls: int, class_names: Dict[int, str]):
        self.pose.bbox.cls = cls
        name = class_names.get(cls, str(cls))
        self.label_item.setText(name)
        # Re-center label if needed
        self.label_item.setPos(self.rect_item.rect().left(), self.rect_item.rect().top() - 20)

    def rotate_pose(self, angle_delta: float):
        self.rotation_angle += angle_delta
        center = self.rect_item.rect().center()
        t = QtGui.QTransform().translate(center.x(), center.y()).rotate(self.rotation_angle).translate(-center.x(), -center.y())
        self.setTransform(t)

    def get_pose(self) -> PoseInstance:
        expected = 5
        win = self.window()
        if hasattr(win, '_expected_kpts'):
            attr = getattr(win, '_expected_kpts')
            if attr is not None:
                expected = int(attr)

        existing_map = {ki.id: ki for ki in self.kpt_items}

        kpts = []
        for i in range(expected):
            if i in existing_map:
                ki = existing_map[i]
                p = ki.pos()
                kpts.append(Keypoint(x=p.x(), y=p.y(), v=ki.v, conf=None))
            else:
                kpts.append(Keypoint(x=0.0, y=0.0, v=0, conf=None))
        
        r = self.rect_item.rect()
        scene_tl = self.mapToScene(r.topLeft())
        scene_br = self.mapToScene(r.bottomRight())
        br = Box(scene_tl.x(), scene_tl.y(), scene_br.x(), scene_br.y(), cls=self.pose.bbox.cls, conf=self.pose.bbox.conf)
        return PoseInstance(bbox=br, kpts=kpts)

    def apply_keypoints(self, kpts: List[Keypoint]) -> None:
        # Update existing keypoint items in-place.
        n = min(len(self.kpt_items), len(kpts))
        for i in range(n):
            kp = kpts[i]
            ki = self.kpt_items[i]
            ki.setPos(float(kp.x), float(kp.y))
            ki.set_visible_state(int(kp.v))
        # If there are extra keypoints, create items.
        for i in range(n, len(kpts)):
            kp = kpts[i]
            ki = KeypointItem(float(kp.x), float(kp.y), i, self)
            ki.set_visible_state(int(kp.v))
            ki.setZValue(20)
            self.kpt_items.append(ki)
            if self.scene():
                self.scene().addItem(ki)
        self.update_geometry()

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
        self._skip_existing = True
        self._delete_rejected = False
        self._images = []
        self._idx = 0
        self._candidates = []
        self._initial_candidates: List[List[PoseInstance]] = []
        self._undo_stack: List[List[PoseInstance]] = []
        self._busy = False
        self._class_names = {}
        self._skeleton = None
        self._expected_kpts = 5
        
        self._delete_confirm_item = None
        self._delete_confirm_time = 0
        self._overwrite_target: Optional[PoseItem] = None

        self._phase_t0 = time.perf_counter()
        self._phase_name = "idle"
        self._phase_base_msg = ""
        self._phase_timer: Optional[QtCore.QTimer] = None
        self._placeholder_item: Optional[QtWidgets.QGraphicsTextItem] = None
        # When True, phase updates can replace the canvas with a placeholder.
        # Turn this on only during initial session/model load.
        self._phase_render_canvas = False

        self._status_progress: Optional[QtWidgets.QProgressBar] = None
        
        self._init_ui()

        # Keep elapsed time moving even if worker emits no new status.
        self._phase_timer = QtCore.QTimer(self)
        self._phase_timer.setInterval(200)
        self._phase_timer.timeout.connect(self._on_phase_tick)
        self._phase_timer.start()

        # Populate default variants.
        self._variants = load_variants(None)
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        for v in self._variants:
            self.variant_combo.addItem(str(v.get("name", "variant")))
        self.variant_combo.setCurrentIndex(0)
        self.variant_combo.blockSignals(False)

        self._init_worker()
        self._install_shortcuts()
        self._load_settings()
        self._apply_lang()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(260)
        self.sidebar_layout = QtWidgets.QVBoxLayout(sidebar)

        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sidebar_scroll.setMinimumWidth(280)
        sidebar_scroll.setMaximumWidth(520)
        splitter.addWidget(sidebar_scroll)
        
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
        
        self.class_list = QtWidgets.QListWidget()
        self.class_list.setFixedHeight(100)
        self.class_list.currentRowChanged.connect(self._on_class_changed)
        self.class_label = QtWidgets.QLabel("Current Class")
        form.addRow(self.class_label, self.class_list)
        
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

        self.rejected_edit = QtWidgets.QLineEdit()
        self.rejected_btn = QtWidgets.QPushButton("...")
        self.rejected_btn.clicked.connect(self._pick_rejected)
        self.rejected_label = QtWidgets.QLabel("Rejected Folder")
        form.addRow(self.rejected_label, self._hbox(self.rejected_edit, self.rejected_btn))
        
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

        self.skip_chk = QtWidgets.QCheckBox("Skip existing output labels")
        self.skip_chk.setChecked(True)
        self.sidebar_layout.addWidget(self.skip_chk)

        self.delete_chk = QtWidgets.QCheckBox("Delete rejected input images (DANGEROUS)")
        self.delete_chk.setChecked(False)
        self.sidebar_layout.addWidget(self.delete_chk)
        
        self.variant_combo = QtWidgets.QComboBox()
        self.variant_combo.currentIndexChanged.connect(self._reload_current)
        self.variant_label = QtWidgets.QLabel("Variant")
        form.addRow(self.variant_label, self.variant_combo)
        
        self.cand_combo = QtWidgets.QComboBox()
        self.cand_combo.currentIndexChanged.connect(self._display_current_candidate)
        self.pick_candidate_label = QtWidgets.QLabel("Candidate")
        form.addRow(self.pick_candidate_label, self.cand_combo)

        self.num_candidates_spin = QtWidgets.QSpinBox()
        self.num_candidates_spin.setRange(1, 20)
        self.num_candidates_spin.setValue(6)
        self.num_candidates_spin.valueChanged.connect(self._reload_current)
        form.addRow(QtWidgets.QLabel("Candidates"), self.num_candidates_spin)

        self.kpt_std_spin = QtWidgets.QDoubleSpinBox()
        self.kpt_std_spin.setRange(0.0, 50.0)
        self.kpt_std_spin.setDecimals(1)
        self.kpt_std_spin.setSingleStep(0.5)
        self.kpt_std_spin.setValue(2.0)
        self.kpt_std_spin.valueChanged.connect(self._reload_current)
        form.addRow(QtWidgets.QLabel("Kpt Jitter (px)"), self.kpt_std_spin)

        self.bbox_std_spin = QtWidgets.QDoubleSpinBox()
        self.bbox_std_spin.setRange(0.0, 50.0)
        self.bbox_std_spin.setDecimals(1)
        self.bbox_std_spin.setSingleStep(0.5)
        self.bbox_std_spin.setValue(1.0)
        self.bbox_std_spin.valueChanged.connect(self._reload_current)
        form.addRow(QtWidgets.QLabel("BBox Jitter (px)"), self.bbox_std_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        self.seed_spin.valueChanged.connect(self._reload_current)
        form.addRow(QtWidgets.QLabel("Seed"), self.seed_spin)

        self.reroll_btn = QtWidgets.QPushButton("Re-roll")
        self.reroll_btn.clicked.connect(lambda: self.seed_spin.setValue(int(self.seed_spin.value()) + 1))
        self.sidebar_layout.addWidget(self.reroll_btn)

        # Device selector
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU0", "cuda:0")
        self.device_combo.currentIndexChanged.connect(self._request_restart_worker)
        self.device_label = QtWidgets.QLabel("Device")
        form.addRow(self.device_label, self.device_combo)

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
        self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1200])
        
        self.status = self.statusBar()

        self._status_progress = QtWidgets.QProgressBar()
        self._status_progress.setRange(0, 0)  # indeterminate
        self._status_progress.setTextVisible(False)
        self._status_progress.setFixedWidth(140)
        self._status_progress.hide()
        self.status.addPermanentWidget(self._status_progress)

    def _set_loading(self, on: bool, msg: str = "") -> None:
        if msg:
            self.status.showMessage(str(msg))
        if self._status_progress is None:
            return
        if on:
            self._status_progress.show()
        else:
            self._status_progress.hide()
            self._phase_base_msg = ""

    def _on_phase_tick(self) -> None:
        if self._status_progress is None or (not self._status_progress.isVisible()):
            return
        if not self._phase_base_msg:
            return
        elapsed = time.perf_counter() - float(self._phase_t0)
        if elapsed < 0:
            elapsed = 0.0
        label = f"{self._phase_base_msg}  ({elapsed:.1f}s)"
        self.status.showMessage(label)
        # Also update the on-canvas placeholder so users see progress.
        try:
            if self._placeholder_item is not None and self._placeholder_item.scene() is not None:
                self._placeholder_item.setPlainText(label)
                br = self._placeholder_item.boundingRect()
                self.scene.setSceneRect(
                    0,
                    0,
                    max(800, int(br.width() + 60)),
                    max(600, int(br.height() + 60)),
                )
                self._placeholder_item.setPos(
                    (self.scene.sceneRect().width() - br.width()) / 2.0,
                    (self.scene.sceneRect().height() - br.height()) / 2.0,
                )
        except Exception:
            pass

    def _set_phase(self, phase: str, msg: str) -> None:
        self._phase_name = str(phase)
        self._phase_t0 = time.perf_counter()
        self._phase_base_msg = str(msg)
        label = f"{msg}  (0.0s)"
        self._set_loading(True, label)
        if self._phase_render_canvas:
            # Only use placeholder rendering during initial load.
            if self._placeholder_item is None or self._placeholder_item.scene() is None:
                self._render_placeholder(label)
            else:
                self._placeholder_item.setPlainText(label)

    def _render_placeholder(self, msg: str) -> None:
        # Don't clear the scene while user is actively drawing.
        try:
            if getattr(self.view, "_temp_rect", None) is not None or getattr(self.view, "_temp_line", None) is not None:
                self.status.showMessage(str(msg))
                return
        except Exception:
            pass
        self.scene.clear()
        t = self.scene.addText(str(msg))
        self._placeholder_item = t
        t.setDefaultTextColor(QtGui.QColor(230, 230, 230))
        f = t.font()
        f.setPointSize(14)
        t.setFont(f)
        t.setZValue(0)
        # center in view
        br = t.boundingRect()
        self.scene.setSceneRect(0, 0, max(800, int(br.width() + 60)), max(600, int(br.height() + 60)))
        t.setPos((self.scene.sceneRect().width() - br.width()) / 2.0, (self.scene.sceneRect().height() - br.height()) / 2.0)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _display_image_only(self) -> None:
        self._phase_render_canvas = False
        self._placeholder_item = None
        self.scene.clear()
        if not self._images:
            return
        img_path = self._images[self._idx]
        pix = QtGui.QPixmap(str(img_path))
        if pix.isNull():
            self._toast(f"Failed to load image: {img_path} (suffix={img_path.suffix})")
            self._render_placeholder("Image load failed")
            return
        img_item = self.scene.addPixmap(pix)
        img_item.setZValue(-100)
        self.scene.setSceneRect(pix.rect())
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.status.showMessage(f"Image {self._idx + 1}/{len(self._images)}: {img_path.name}")

    def _remove_pose_from_current_candidate(self, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        if not self._candidates or self.cand_combo.currentIndex() < 0:
            return
        idx = int(self.cand_combo.currentIndex())
        if not (0 <= idx < len(self._candidates)):
            return

        x1, y1, x2, y2 = bbox_xyxy

        def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0.0:
                return 0.0
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            u = a_area + b_area - inter
            return 0.0 if u <= 0.0 else float(inter / u)

        target = (float(x1), float(y1), float(x2), float(y2))
        best_i: Optional[int] = None
        best = 0.0
        for i, p in enumerate(self._candidates[idx]):
            try:
                b = p.bbox
                cand = (float(b.x1), float(b.y1), float(b.x2), float(b.y2))
                sc = _iou(cand, target)
                if sc > best:
                    best = sc
                    best_i = int(i)
            except Exception:
                continue
        if best_i is not None:
            try:
                self._candidates[idx].pop(best_i)
            except Exception:
                pass

    def _on_bbox_drawn(self, rect: QtCore.QRectF):
        if not self._images or self.cand_combo.currentIndex() < 0: return

        self._push_undo()
        
        cls_idx = max(0, self.class_list.currentRow())
        
        # If overwrite is active, do a region inference to regenerate keypoints.
        if self._overwrite_target is not None and self._worker_ready and (not self._busy):
            old = self._overwrite_target
            self._overwrite_target = None

            try:
                op = old.get_pose()
                self._remove_pose_from_current_candidate((op.bbox.x1, op.bbox.y1, op.bbox.x2, op.bbox.y2))
            except Exception:
                pass

            # Remove old instance now (overwrite semantics).
            try:
                if old.scene():
                    old.scene().removeItem(old)
                for k in list(getattr(old, "kpt_items", [])):
                    if k.scene():
                        k.scene().removeItem(k)
                for l in list(getattr(old, "skeleton_lines", [])):
                    if l.scene():
                        l.scene().removeItem(l)
            except Exception:
                pass

            # Request region inference from worker.
            img_path = self._images[self._idx]
            req_id = int(self._next_req_id)
            self._next_req_id += 1
            self._inflight_id = req_id
            self._inflight_kind = "infer_region"
            self._busy = True

            self._pending_region = {
                "mode": "overwrite",
                "rect": rect,
                "cls": cls_idx,
            }
            msg = {
                "type": "infer_region",
                "id": req_id,
                "image": str(img_path),
                "bbox": [float(rect.left()), float(rect.top()), float(rect.right()), float(rect.bottom())],
                "expand": 0.2,
                "conf": float(self.conf_spin.value()),
                "iou": float(self.iou_spin.value()),
                "expected_kpts": int(self._expected_kpts) if self._expected_kpts is not None else None,
            }
            self._worker.write((json.dumps(msg) + "\n").encode("utf-8"))
            self._toast("Overwrite relabel: running local inference...")
            return

        # Fallback: manual keypoint seed.
        k = self._expected_kpts if self._expected_kpts is not None else 5
        kpts: List[Keypoint] = []
        cx, cy = rect.center().x(), rect.center().y()
        rw, rh = rect.width(), rect.height()
        if int(k) == 5:
            pts = [
                (cx, cy),
                (rect.left(), rect.top()),
                (rect.right(), rect.top()),
                (rect.left(), rect.bottom()),
                (rect.right(), rect.bottom()),
            ]
            for px, py in pts:
                kpts.append(Keypoint(float(px), float(py), v=2, conf=None))
        else:
            cols = int(np.ceil(np.sqrt(int(k))))
            rows = int(np.ceil(int(k) / cols))
            for i in range(int(k)):
                r = i // cols
                c = i % cols
                px = rect.left() + (c + 0.5) * (rw / cols)
                py = rect.top() + (r + 0.5) * (rh / rows)
                kpts.append(Keypoint(float(px), float(py), v=2, conf=None))

        new_bbox = Box(rect.left(), rect.top(), rect.right(), rect.bottom(), cls=cls_idx, conf=1.0)
        new_pose = PoseInstance(bbox=new_bbox, kpts=kpts)
        if self._candidates is not None:
            self._candidates[self.cand_combo.currentIndex()].append(new_pose)
        item = PoseItem(new_pose, self._class_names, self._skeleton)
        item.add_to_scene(self.scene)
        item.setSelected(True)
        self._toast(f"New {self._class_names.get(cls_idx, cls_idx)} pose created")

    def _on_class_changed(self, row: int):
        if row < 0: return
        # If an item is selected, update its class immediately
        selected = self.scene.selectedItems()
        if selected:
            item = selected[0]
            target = item if isinstance(item, PoseItem) else getattr(item, 'parent_pose', None)
            if target:
                target.set_class(row, self._class_names)
                self.status.showMessage(f"Updated class to {self._class_names.get(row, row)}", 2000)

    def _init_worker(self):
        self._worker = QtCore.QProcess(self)
        # IMPORTANT: ultralytics/torch may write to stderr during import/init.
        # If stderr is not drained, the child process can block when the pipe buffer fills.
        # Merge channels so we always drain output from a single reader.
        self._worker.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._worker.readyReadStandardOutput.connect(self._on_worker_stdout)
        self._worker_ready = False
        self._worker_buf = ""
        self._restart_requested = False
        self._next_req_id = 1
        self._inflight_id: Optional[int] = None
        self._inflight_kind: Optional[str] = None
        # infer_region pending request:
        # - mode: "overwrite" | "refine"
        # - rect: QRectF (overwrite)
        # - pose_item: PoseItem (refine)
        # - cls: int
        self._pending_region: Optional[Dict[str, Any]] = None

    def _request_restart_worker(self):
        # Schedule restart after current busy task
        self._restart_requested = True
        if not self._busy:
            self._restart_worker()

    def _restart_worker(self):
        self._restart_requested = False
        if self._worker.state() != QtCore.QProcess.ProcessState.NotRunning:
            self._worker.terminate()
            if not self._worker.waitForFinished(1000):
                self._worker.kill()
                self._worker.waitForFinished(500)
        
        py = sys.executable
        script = str((Path(__file__).parent / "worker" / "infer_worker.py").resolve())
        self._worker.setProgram(py)
        self._worker.setArguments([script])
        self._worker.setWorkingDirectory(str(Path(__file__).parent))

        self._phase_t0 = time.time()
        self._set_phase("starting_worker", "Starting worker...")
        self._worker.start()
        if self._worker.waitForStarted(5000):
            self._set_phase("worker_started", "Worker started. Importing deps...")
            dev = "cpu"
            if hasattr(self, "device_combo"):
                dev = str(self.device_combo.currentData())
            init = {"type": "init", "model": str(self._model_path), "device": dev}
            self._worker.write((json.dumps(init) + "\n").encode("utf-8"))
        else:
            self._set_loading(False)
            self._toast("Worker failed to start")
            self._render_placeholder("Worker failed to start")

    def _on_worker_stdout(self):
        raw_data = self._worker.readAllStandardOutput()
        data = bytes(raw_data.data()).decode("utf-8", errors="replace")
        self._worker_buf += data
        while "\n" in self._worker_buf:
            line, self._worker_buf = self._worker_buf.split("\n", 1)
            try:
                msg = json.loads(line)
                if msg.get("type") == "status":
                    phase = str(msg.get("phase", ""))
                    if phase == "import_ultralytics":
                        self._set_phase(phase, "Importing Ultralytics/Torch...")
                    elif phase == "loading_model":
                        self._set_phase(phase, "Loading model weights...")
                    elif phase == "loading_image":
                        self._set_phase(phase, "Loading image & running inference...")
                    else:
                        self._set_phase(phase or "working", "Working...")
                    continue
                if msg.get("type") == "ready":
                    self._worker_ready = True
                    self._phase_t0 = time.time()
                    self._set_phase("ready", "Model ready. Running inference...")
                    self._reload_current()
                elif msg.get("type") == "result":
                    rid = msg.get("id")
                    kind = msg.get("kind", "infer")
                    if self._inflight_id is not None and rid is not None and int(rid) != int(self._inflight_id):
                        continue

                    self._busy = False
                    self._inflight_id = None
                    self._inflight_kind = None

                    if msg.get("ok"):
                        self._set_loading(False)
                        if str(kind) == "infer_region":
                            self._handle_infer_region_result(msg["payload"])
                        else:
                            self._handle_result(msg["payload"])
                    else:
                        self._set_loading(False)
                        self._toast(msg.get("error"))
                        if str(kind) == "infer_region":
                            self._handle_infer_region_failed()
                        else:
                            # Show the image even if inference failed.
                            self._display_image_only()
                    
                    if self._restart_requested:
                        self._restart_worker()
            except: pass

    def _handle_infer_region_failed(self) -> None:
        if not self._pending_region:
            return
        mode = str(self._pending_region.get("mode", "overwrite"))
        if mode == "overwrite":
            # Fallback: create a manual pose in the last requested rect.
            try:
                rect = self._pending_region.get("rect")
                cls_idx = int(self._pending_region.get("cls", 0))
                if isinstance(rect, QtCore.QRectF):
                    self._pending_region = None
                    self._on_bbox_drawn(rect)
                    # restore desired class
                    sel = self.scene.selectedItems()
                    if sel:
                        it0 = sel[0]
                        pose = it0 if isinstance(it0, PoseItem) else getattr(it0, "parent_pose", None)
                        if isinstance(pose, PoseItem):
                            pose.set_class(cls_idx, self._class_names)
            except Exception:
                self._pending_region = None
        else:
            # refine mode: keep current annotations
            self._pending_region = None
            self._toast("Predict failed")

    def _handle_infer_region_result(self, payload: Dict[str, Any]) -> None:
        if not self._pending_region:
            return

        mode = str(self._pending_region.get("mode", "overwrite"))

        def _parse_kpts(pd: Dict[str, Any]) -> List[Keypoint]:
            expected = int(self._expected_kpts) if self._expected_kpts is not None else 5
            kpts_raw = pd.get("kpts", [])
            kpts: List[Keypoint] = []
            for k in kpts_raw:
                if not (isinstance(k, list) and len(k) >= 3):
                    continue
                kpts.append(Keypoint(float(k[0]), float(k[1]), v=int(k[2]), conf=(None if len(k) < 4 else k[3])))
            if len(kpts) < expected:
                kpts = kpts + [Keypoint(0.0, 0.0, v=0, conf=None) for _ in range(expected - len(kpts))]
            elif len(kpts) > expected:
                kpts = kpts[:expected]
            return kpts

        try:
            pd = payload.get("pose", {})
            if mode == "refine":
                pose_item = self._pending_region.get("pose_item")
                cls_idx = int(self._pending_region.get("cls", 0))
                if not isinstance(pose_item, PoseItem):
                    return
                kpts = _parse_kpts(pd)
                pose_item.set_class(cls_idx, self._class_names)
                pose_item.apply_keypoints(kpts)
                self._toast("Predict applied")
                return

            # overwrite mode (existing behavior): create new pose for drawn rect
            rect = self._pending_region.get("rect")
            cls_idx = int(self._pending_region.get("cls", 0))
            if not isinstance(rect, QtCore.QRectF):
                return
            bbox = Box(float(rect.left()), float(rect.top()), float(rect.right()), float(rect.bottom()), cls=cls_idx, conf=pd.get("conf", None))
            kpts = _parse_kpts(pd)
            new_pose = PoseInstance(bbox=bbox, kpts=kpts)
            if self._candidates is not None and self.cand_combo.currentIndex() >= 0:
                self._candidates[self.cand_combo.currentIndex()].append(new_pose)
            item = PoseItem(new_pose, self._class_names, self._skeleton)
            item.add_to_scene(self.scene)
            item.setSelected(True)
            self._toast("Overwrite relabel applied")
        finally:
            self._pending_region = None

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

        # Snapshot initial candidates for reset/undo.
        self._initial_candidates = [copy_pose_instances(c) for c in self._candidates]
        self._undo_stack = []
        self.cand_combo.blockSignals(True)
        self.cand_combo.clear()
        for i in range(len(self._candidates)):
            self.cand_combo.addItem(str(i))
        self.cand_combo.setCurrentIndex(0)
        self.cand_combo.blockSignals(False)
        self._display_current_candidate()

    def _display_current_candidate(self):
        # Don't show a partially-initialized view; use progress/placeholder until inference returns.
        if (not self._candidates) and (self._busy or not self._worker_ready):
            self._phase_render_canvas = True
            self._render_placeholder("Loading...")
            return

        self._phase_render_canvas = False
        self._placeholder_item = None

        self.scene.clear()
        if not self._images:
            return
        img_path = self._images[self._idx]
        pix = QtGui.QPixmap(str(img_path))
        if pix.isNull():
            self._toast(f"Failed to load image: {img_path} (suffix={img_path.suffix})")
            self._render_placeholder("Image load failed")
            return
        img_item = self.scene.addPixmap(pix)
        img_item.setZValue(-100)
        self.scene.setSceneRect(pix.rect())

        if self._candidates and self.cand_combo.currentIndex() >= 0:
            poses = self._candidates[self.cand_combo.currentIndex()]
            for p in poses:
                item = PoseItem(p, self._class_names, self._skeleton)
                item.add_to_scene(self.scene)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.status.showMessage(f"Image {self._idx + 1}/{len(self._images)}: {img_path.name}")

    def _cycle_candidate_by_wheel(self, delta_y: int) -> None:
        if self.cand_combo.count() <= 0:
            return
        step = 1 if delta_y > 0 else -1
        i = int(self.cand_combo.currentIndex())
        n = int(self.cand_combo.count())
        self.cand_combo.setCurrentIndex((i + step) % n)

    def _push_undo(self) -> None:
        if not self._candidates or self.cand_combo.currentIndex() < 0:
            return
        poses = self._snapshot_scene_poses()
        self._undo_stack.append(copy_pose_instances(poses))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        poses = self._undo_stack.pop()
        self._set_current_candidate_poses(poses)
        self._toast("Undo")

    def _reset_current_candidate(self) -> None:
        if not self._initial_candidates or self.cand_combo.currentIndex() < 0:
            return
        idx = int(self.cand_combo.currentIndex())
        if 0 <= idx < len(self._initial_candidates):
            self._set_current_candidate_poses(copy_pose_instances(self._initial_candidates[idx]))
            self._toast("Reset")

    def _snapshot_scene_poses(self) -> List[PoseInstance]:
        return [item.get_pose() for item in self.scene.items() if isinstance(item, PoseItem)]

    def _set_current_candidate_poses(self, poses: List[PoseInstance]) -> None:
        if not self._candidates or self.cand_combo.currentIndex() < 0:
            return
        idx = int(self.cand_combo.currentIndex())
        if 0 <= idx < len(self._candidates):
            self._candidates[idx] = poses
        self._display_current_candidate()

    def _out_label_path_for(self, image_path: Path) -> Path:
        out_dir = Path(self.output_edit.text().strip()) if self.output_edit.text().strip() else Path("output")
        return out_dir / "labels" / f"{image_path.stem}.txt"

    def _skip_to_next_unlabeled(self, direction: int) -> None:
        if not self._images:
            return
        if not bool(self.skip_chk.isChecked()):
            return
        n = len(self._images)
        for _ in range(n):
            p = self._images[self._idx]
            if p.exists() and not self._out_label_path_for(p).exists():
                return
            self._idx = (self._idx + direction) % n

    def _load_session(self):
        model_txt = self.model_edit.text().strip()
        input_txt = self.input_edit.text().strip()
        if not model_txt or not input_txt:
            self._toast("Please select model and input folder")
            return

        model_path = Path(model_txt)
        input_dir = Path(input_txt)
        output_dir = Path(self.output_edit.text().strip()) if self.output_edit.text().strip() else Path("output")
        rejected_dir = Path(self.rejected_edit.text().strip()) if self.rejected_edit.text().strip() else (output_dir / "rejected")

        if not model_path.exists():
            self._toast("Model path not found")
            return
        if not input_dir.exists():
            self._toast("Input folder not found")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

        self._model_path = model_path
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._rejected_dir = rejected_dir
        self._skip_existing = bool(self.skip_chk.isChecked())
        self._delete_rejected = bool(self.delete_chk.isChecked())

        self._set_loading(True, "Scanning images...")
        QtWidgets.QApplication.processEvents()
        self._images = list_images(self._input_dir)
        if not self._images:
            self._set_loading(False)
            self._toast("No images found")
            return

        self._idx = 0
        self._skip_to_next_unlabeled(+1)

        # Start worker + show loading indicator. We only display image after inference.
        self._candidates = []
        self._initial_candidates = []
        self._undo_stack = []
        self.cand_combo.blockSignals(True)
        self.cand_combo.clear()
        self.cand_combo.blockSignals(False)
        self._phase_render_canvas = True
        self._render_placeholder("Loading model...")
        self._set_loading(True, "Loading model...")

        self._restart_worker()

    def _reload_current(self):
        if not self._images or self._busy:
            return
        if not self._worker_ready:
            return

        img_path = self._images[self._idx]
        if not img_path.exists():
            self._next()
            return

        self._busy = True
        self._set_loading(True, "Running inference...")

        v_idx = int(self.variant_combo.currentIndex())
        v = self._variants[v_idx] if 0 <= v_idx < len(self._variants) else (self._variants[0] if self._variants else {})

        req_id = int(self._next_req_id)
        self._next_req_id += 1
        self._inflight_id = req_id
        self._inflight_kind = "infer"

        msg = {
            "type": "infer",
            "id": req_id,
            "image": str(img_path),
            "conf": float(self.conf_spin.value()),
            "iou": float(self.iou_spin.value()),
            "variant_json": json.dumps(v),
            "seed": int(self.seed_spin.value()),
            "num_candidates": int(self.num_candidates_spin.value()),
            "kpt_std_px": float(self.kpt_std_spin.value()),
            "bbox_std_px": float(self.bbox_std_spin.value()),
        }
        self._worker.write((json.dumps(msg) + "\n").encode("utf-8"))

    def _predict_selected_bbox(self) -> None:
        # Predict keypoints around the selected pose bbox (with expansion).
        if self._busy or (not self._worker_ready):
            return
        sel = self.scene.selectedItems()
        if not sel:
            self._toast("Select a pose first")
            return
        it0 = sel[0]
        pose_item = it0 if isinstance(it0, PoseItem) else getattr(it0, "parent_pose", None)
        if not isinstance(pose_item, PoseItem):
            self._toast("Select a pose first")
            return
        if not self._images:
            return

        self._push_undo()

        p = pose_item.get_pose()
        b = p.bbox
        cls_idx = int(b.cls)

        img_path = self._images[self._idx]
        req_id = int(self._next_req_id)
        self._next_req_id += 1
        self._inflight_id = req_id
        self._inflight_kind = "infer_region"
        self._busy = True

        self._pending_region = {
            "mode": "refine",
            "pose_item": pose_item,
            "cls": cls_idx,
        }

        msg = {
            "type": "infer_region",
            "id": req_id,
            "image": str(img_path),
            "bbox": [float(b.x1), float(b.y1), float(b.x2), float(b.y2)],
            "expand": 0.2,
            "conf": float(self.conf_spin.value()),
            "iou": float(self.iou_spin.value()),
            "expected_kpts": int(self._expected_kpts) if self._expected_kpts is not None else None,
        }
        self._worker.write((json.dumps(msg) + "\n").encode("utf-8"))
        self._toast("Predicting in bbox...")

    def _next(self):
        if self._images:
            self._idx = (self._idx + 1) % len(self._images)
            self._skip_to_next_unlabeled(+1)
            self._reload_current()

    def _prev(self):
        if self._images:
            self._idx = (self._idx - 1) % len(self._images)
            self._skip_to_next_unlabeled(-1)
            self._reload_current()

    def _accept(self):
        if not self._images or not self._output_dir: return
        
        # Sync and filter: only keep objects that have at least one visible point
        raw_poses = [item.get_pose() for item in self.scene.items() if isinstance(item, PoseItem)]
        current_poses = [p for p in raw_poses if any(k.v > 0 for k in p.kpts)]
        
        img_path = self._images[self._idx]
        img_raw = cv2.imread(str(img_path))
        if img_raw is not None:
            # save_sample will write empty txt if current_poses is empty
            save_sample(img_path, img_raw, current_poses, self._output_dir / "images", self._output_dir / "labels")
            # Keep candidate cache in sync with edits.
            if self._candidates and self.cand_combo.currentIndex() >= 0:
                self._candidates[self.cand_combo.currentIndex()] = copy_pose_instances(current_poses)
        self._next()

    def _reject(self):
        if not self._images or self._rejected_dir is None:
            self._next()
            return
        img_path = self._images[self._idx]
        try:
            safe_reject(img_path, self._rejected_dir, hard_delete=bool(self.delete_chk.isChecked()))
        except Exception as e:
            self._toast(str(e))
        self._next()

    def _clear_all_marks(self):
        self._push_undo()
        self._set_current_candidate_poses([])

    def _pick_line_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.view.set_line_color(color)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Undo / reset
        if (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) and event.key() == QtCore.Qt.Key.Key_Z:
            self._undo()
            return
        if (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) and event.key() == QtCore.Qt.Key.Key_R:
            self._reset_current_candidate()
            return

        # Digit class switching (1-9)
        if QtCore.Qt.Key.Key_1 <= event.key() <= QtCore.Qt.Key.Key_9:
            idx = int(event.key() - QtCore.Qt.Key.Key_1)
            if 0 <= idx < self.class_list.count():
                self.class_list.setCurrentRow(idx)
                return

        # Cycle selection under cursor.
        if event.key() == QtCore.Qt.Key.Key_Tab:
            try:
                view_pos = self.view.mapFromGlobal(QtGui.QCursor.pos())
                scene_pos = self.view.mapToScene(view_pos)
                items = self.scene.items(scene_pos)
                poses: List[PoseItem] = []
                for it in items:
                    if isinstance(it, PoseItem):
                        poses.append(it)
                if poses:
                    # Deterministic ordering.
                    poses = sorted(poses, key=lambda p: float(p.zValue()), reverse=True)
                    cur = None
                    sel = self.scene.selectedItems()
                    if sel:
                        cur0 = sel[0]
                        cur = cur0 if isinstance(cur0, PoseItem) else getattr(cur0, "parent_pose", None)
                    if cur in poses:
                        i = poses.index(cur)
                        nxt = poses[(i + 1) % len(poses)]
                    else:
                        nxt = poses[0]
                    self.scene.clearSelection()
                    nxt.setSelected(True)
                    return
            except Exception:
                pass

        if event.key() == QtCore.Qt.Key.Key_Delete or event.key() == QtCore.Qt.Key.Key_Backspace or event.key() == QtCore.Qt.Key.Key_X:
            selected = self.scene.selectedItems()
            if selected:
                self._push_undo()
                item = selected[0]
                if isinstance(item, KeypointItem):
                    p = item.parent_pose
                    expected = self._expected_kpts if self._expected_kpts is not None else 5
                    visible_count = len([k for k in p.kpt_items if k.v > 0])
                    if visible_count > expected:
                        if item in p.kpt_items: p.kpt_items.remove(item)
                        self.scene.removeItem(item)
                        self._toast("Keypoint removed")
                    else:
                        item.set_visible_state(0)
                        self._toast("Keypoint hidden")
                    p.update_geometry()
                elif isinstance(item, GuideLineItem):
                    self.scene.removeItem(item)
                    self._toast("Guide line removed")
                else:
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
        self.class_label.setText("当前类别" if zh else "Current Class")
        self.input_label.setText("输入文件夹" if zh else "Input Folder")
        self.output_label.setText("输出文件夹" if zh else "Output Folder")
        self.rejected_label.setText("拒绝文件夹" if zh else "Rejected Folder")
        self.skip_chk.setText("跳过已输出标签" if zh else "Skip existing output labels")
        self.delete_chk.setText("拒绝时删除输入图片(危险)" if zh else "Delete rejected input images (DANGEROUS)")
        self.load_btn.setText("启动会话" if zh else "Load Session")
        self.accept_btn.setText("保存 (S)" if zh else "Save (S)")
        self.reject_btn.setText("拒绝 (R)" if zh else "Reject (R)")
        self.clear_btn.setText("清除标注 (Del)" if zh else "Clear Marks (Del)")
        self.create_mode_chk.setText("绘框模式 (N)" if zh else "Create Pose Mode (N)")
        self.line_mode_chk.setText("画辅助线模式 (L)" if zh else "Draw Line Mode (L)")
        self.color_btn.setText("选取线条颜色" if zh else "Pick Line Color")
        self.reroll_btn.setText("重新随机" if zh else "Re-roll")
        self.shortcuts_label.setText(
            "快捷键:\nA/D: 上/下一张\nS/R: 保存/拒绝\n1-9: 切换类别\nDel/X: 隐藏点 / 删线\n右键: 快隐点 / 快删线\n双击Del: 删整框\nN: 绘框模式\nShift+N: 补缺失点\nAlt+N: 新增关键点\nL: 画线模式\nCtrl+滚轮: 缩放\n空格+拖拽: 平移\n[/]: 旋转框" if zh else
            "Shortcuts:\nA/D: Prev/Next\nS/R: Save/Reject\n1-9: Switch Class\nDel/X: Hide point / Del line\nRight-Click: Fast hide/del\nDouble-Del: Remove bbox\nN: Create mode\nShift+N: Add missing kpt\nAlt+N: Add new kpt\nL: Line mode\nCtrl+Wheel: Zoom\nSpace+Drag: Pan\n[/]: Rotate"
        )




    def _pick_project_yaml(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YAML", "", "YAML (*.yaml *.yml)")
        if path:
            self.project_yaml_edit.setText(path)
            data = load_project_yaml(Path(path))
            self._class_names = data["names"]
            self._skeleton = data["skeleton"]
            self._expected_kpts = data.get("expected_kpts", 5)
            
            # Populate class list
            self.class_list.blockSignals(True)
            self.class_list.clear()
            for i in sorted(self._class_names.keys()):
                self.class_list.addItem(f"{i}: {self._class_names[i]}")
            self.class_list.setCurrentRow(0)
            self.class_list.blockSignals(False)

    def _pick_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", "", "Model (*.pt)")
        if path: self.model_edit.setText(path)

    def _pick_input(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path: self.input_edit.setText(path)

    def _pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_edit.setText(path)
            if not self.rejected_edit.text().strip():
                self.rejected_edit.setText(str(Path(path) / "rejected"))

    def _pick_rejected(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Rejected Folder")
        if path:
            self.rejected_edit.setText(path)

    def _toast(self, msg):
        self.status.showMessage(str(msg), 3000)

    def _snap_kpt_to_guides(self, value: QtCore.QPointF) -> QtCore.QPointF:
        # Best-effort snapping: if a keypoint is close to a guide line, project onto it.
        try:
            x0 = float(value.x())
            y0 = float(value.y())
            best = None
            best_d2 = float(8.0 * 8.0)
            for it in self.scene.items():
                if not isinstance(it, GuideLineItem):
                    continue
                ln = it.line()
                ax, ay = float(ln.x1()), float(ln.y1())
                bx, by = float(ln.x2()), float(ln.y2())
                vx, vy = (bx - ax), (by - ay)
                wx, wy = (x0 - ax), (y0 - ay)
                denom = vx * vx + vy * vy
                if denom <= 1e-9:
                    continue
                t = (wx * vx + wy * vy) / denom
                t = float(max(0.0, min(1.0, t)))
                px = ax + t * vx
                py = ay + t * vy
                d2 = (px - x0) ** 2 + (py - y0) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (px, py)
            if best is not None:
                return QtCore.QPointF(best[0], best[1])
        except Exception:
            pass
        return value

    def _hbox(self, a, b):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0,0,0,0)
        l.addWidget(a)
        l.addWidget(b)
        return w

    def _on_alt_n(self):
        # Add COMPLETELY NEW point at cursor
        selected = self.scene.selectedItems()
        if not selected:
            return
        item = selected[0]
        pose = item if isinstance(item, PoseItem) else getattr(item, 'parent_pose', None)
        if not pose:
            return
            
        new_id = 0
        if pose.kpt_items:
            new_id = max(k.id for k in pose.kpt_items) + 1
        
        # Get scene pos from view's cursor
        view_pos = self.view.mapFromGlobal(QtGui.QCursor.pos())
        scene_pos = self.view.mapToScene(view_pos)
        
        ki = KeypointItem(scene_pos.x(), scene_pos.y(), new_id, pose)
        ki.set_visible_state(2)
        pose.kpt_items.append(ki)
        self.scene.addItem(ki)
        pose.update_geometry()
        self._toast(f"Added new kpt {new_id}")

    def _on_shift_n(self):
        # Revive missing kpt at cursor
        selected = self.scene.selectedItems()
        if not selected:
            return
        item = selected[0]
        pose = item if isinstance(item, PoseItem) else getattr(item, 'parent_pose', None)
        if not pose:
            return
            
        for ki in pose.kpt_items:
            if ki.v == 0:
                view_pos = self.view.mapFromGlobal(QtGui.QCursor.pos())
                scene_pos = self.view.mapToScene(view_pos)
                ki.setPos(scene_pos)
                ki.set_visible_state(2)
                ki.parent_pose.update_geometry()
                self._toast(f"Revived kpt {ki.id}")
                break

    def _install_shortcuts(self):
        def add(k, fn):
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.activated.connect(fn)
        add("D", self._next)
        add("Right", self._next)
        add("Space", self._accept)
        add("A", self._prev)
        add("Left", self._prev)
        add("S", self._accept)
        add("R", self._reject)
        add("L", lambda: self.line_mode_chk.toggle())
        add("Ctrl+L", lambda: self.line_mode_chk.toggle())
        add("C", lambda: self._cycle_candidate_by_wheel(+1))
        add("Shift+N", self._on_shift_n)
        add("Alt+N", self._on_alt_n)
        add("P", self._predict_selected_bbox)

    def _load_settings(self):
        self._settings = QtCore.QSettings("YoloLabeler", "Integrated")
        
        m = self._settings.value("model_path", "")
        if m:
            self.model_edit.setText(str(m))
        
        i = self._settings.value("input_dir", "")
        if i:
            self.input_edit.setText(str(i))
        
        o = self._settings.value("output_dir", "")
        if o:
            self.output_edit.setText(str(o))

        rj = self._settings.value("rejected_dir", "")
        if rj:
            self.rejected_edit.setText(str(rj))

        self.skip_chk.setChecked(bool(self._settings.value("skip_existing", True)))
        self.delete_chk.setChecked(bool(self._settings.value("delete_rejected", False)))

        try:
            self.num_candidates_spin.setValue(int(str(self._settings.value("num_candidates", 6))))
            self.seed_spin.setValue(int(str(self._settings.value("seed", 0))))
            self.kpt_std_spin.setValue(float(str(self._settings.value("kpt_std_px", 2.0))))
            self.bbox_std_spin.setValue(float(str(self._settings.value("bbox_std_px", 1.0))))
        except Exception:
            pass
        
        y = self._settings.value("project_yaml", "")
        if y:
            self.project_yaml_edit.setText(str(y))
            # Auto-load yaml if exists
            if Path(str(y)).exists():
                try:
                    data = load_project_yaml(Path(str(y)))
                    self._class_names = data["names"]
                    self._skeleton = data["skeleton"]
                    self._expected_kpts = data.get("expected_kpts", 5)
                    self.class_list.clear()
                    for k in sorted(self._class_names.keys()):
                        self.class_list.addItem(f"{k}: {self._class_names[k]}")
                    self.class_list.setCurrentRow(0)
                except: pass

    def _save_settings(self):
        self._settings.setValue("model_path", self.model_edit.text())
        self._settings.setValue("input_dir", self.input_edit.text())
        self._settings.setValue("output_dir", self.output_edit.text())
        self._settings.setValue("rejected_dir", self.rejected_edit.text())
        self._settings.setValue("project_yaml", self.project_yaml_edit.text())

        self._settings.setValue("skip_existing", bool(self.skip_chk.isChecked()))
        self._settings.setValue("delete_rejected", bool(self.delete_chk.isChecked()))

        self._settings.setValue("num_candidates", int(self.num_candidates_spin.value()))
        self._settings.setValue("seed", int(self.seed_spin.value()))
        self._settings.setValue("kpt_std_px", float(self.kpt_std_spin.value()))
        self._settings.setValue("bbox_std_px", float(self.bbox_std_spin.value()))

    def closeEvent(self, event):
        self._save_settings()
        self._stop_worker()
        super().closeEvent(event)

    def _stop_worker(self):
        if self._worker.state() != QtCore.QProcess.ProcessState.NotRunning:
            self._worker.terminate()
            if not self._worker.waitForFinished(1000):
                self._worker.kill()

def main():

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
