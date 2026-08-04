#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizer — 多臂组合显示窗口
==============================
将所有臂的相机画面 + GGCNN 热力图 + 全局状态信息组合到单个 OpenCV 窗口。

布局 (horizontal 模式):
┌──────────────────────┬──────────────────────┐
│  Arm-0 彩色图        │  Arm-1 彩色图         │
│  + GGCNN热力图       │  + GGCNN热力图        │
├──────────────────────┼──────────────────────┤
│  Arm-2 彩色图        │  全局状态面板          │
│  + GGCNN热力图       │  (状态/计数/区域/距离) │
└──────────────────────┴──────────────────────┘

每秒刷新约 30 帧，独立渲染线程。
按 q/ESC 退出，按 r 清除 HAZARD，按 s 打印系统摘要。
"""

import time
import threading
import logging
import numpy as np
import cv2
from typing import Dict, Optional

from .config import SystemConfig
from .coordinator import MultiArmCoordinator
from .collision_avoidance import interarm_distance

logger = logging.getLogger(__name__)

# 状态对应的显示颜色 (BGR)
STATE_COLORS = {
    "IDLE":        (200, 200, 200),   # 灰色
    "LOCKED":      (0, 255, 255),     # 黄色
    "MOVING":      (255, 165, 0),     # 橙色
    "GRASPING":    (0, 255, 0),       # 绿色
    "PLACING":     (255, 0, 255),     # 品红
    "RECOVERING":  (0, 165, 255),     # 橙红
    "STOPPED":     (0, 0, 255),       # 红色
    "DISCONNECTED":(0, 0, 128),       # 深红
    "INIT":        (128, 128, 128),   # 暗灰
    "NO OBJECT":   (200, 200, 200),
    "SEARCHING":   (255, 200, 100),
    "WAIT_ZONE":   (100, 200, 255),
    "FINE_LOC":    (200, 255, 200),
}


class MultiArmVisualizer:
    """
    多臂组合可视化。

    在独立线程中运行，从各 ArmController 拉取帧数据，
    从 Coordinator 拉取全局状态，拼合成一个显示窗口。
    """

    def __init__(self, config: SystemConfig,
                 coordinator: MultiArmCoordinator):
        self._config = config
        self._coord = coordinator

        # arm_id → ArmController 引用（用于拉取帧）
        self._controllers: Dict[int, object] = {}

        self._window_name = config.win_name
        self._layout = config.display_layout
        self._render_thread: Optional[threading.Thread] = None
        self._running = False

        # HAZARD 闪烁状态
        self._hazard_blink = False
        self._hazard_last_toggle = 0.0

    def attach_controller(self, ctrl):
        """绑定 ArmController 实例（在各臂创建后调用）。"""
        self._controllers[ctrl._cfg.arm_id] = ctrl

    def start(self):
        """启动渲染线程。"""
        if self._running:
            return
        self._running = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 1280, 720)
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="Visualizer",
            daemon=True,
        )
        self._render_thread.start()
        logger.info("Visualizer 已启动")

    def stop(self):
        """停止渲染。"""
        self._running = False

    def join(self, timeout: Optional[float] = None):
        """等待渲染线程结束。"""
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout)

    # ═══════════════════════════════════════════════════════════════════
    # 渲染循环
    # ═══════════════════════════════════════════════════════════════════

    def _render_loop(self):
        """渲染主循环（约 30Hz）。"""
        logger.info("Visualizer 渲染循环开始")
        while self._running and not self._coord.global_stop_event.is_set():
            combined = self._build_combined_frame()
            if combined is not None:
                cv2.imshow(self._window_name, combined)

            key = cv2.waitKey(33) & 0xFF
            if key == ord('q') or key == 27:
                # 全局退出
                self._coord.broadcast_stop()
                break
            elif key == ord('r'):
                # 清除 HAZARD 区域（需操作员确认安全）
                if self._coord.any_hazard():
                    self._coord.clear_hazard()
                    logger.info("[Visualizer] HAZARD 已人工清除")
            elif key == ord('s'):
                # 打印系统摘要
                summary = self._coord.get_summary()
                logger.info(f"[Visualizer] 系统摘要: {summary}")

        cv2.destroyWindow(self._window_name)
        logger.info("Visualizer 停止")

    # ═══════════════════════════════════════════════════════════════════
    # 帧构建
    # ═══════════════════════════════════════════════════════════════════

    def _build_combined_frame(self) -> Optional[np.ndarray]:
        """拼合所有臂的图像 + 全局状态面板。"""
        arm_count = len(self._config.arms)
        if arm_count == 0:
            return None

        # 收集所有臂的最新帧
        frames = {}
        for arm_id, ctrl in sorted(self._controllers.items()):
            color, heatmap, label, count = ctrl.get_latest_frame()
            if color is not None:
                frames[arm_id] = (color, heatmap, label, count)

        if not frames:
            # 还没有收到任何帧 → 显示等待画面
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for cameras...",
                        (120, 250), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            return placeholder

        summary = self._coord.get_summary()
        tile_w, tile_h = 640, 480

        if self._layout == "horizontal":
            return self._build_horizontal(frames, summary, tile_w, tile_h)
        else:
            return self._build_vertical(frames, summary, tile_w, tile_h)

    def _build_horizontal(self, frames, summary, tw, th) -> np.ndarray:
        """水平布局：上半部 Arm-0 + Arm-1，下半部 Arm-2 + 状态面板。"""
        top_row = []
        bottom_row = []

        for arm_id in sorted(frames.keys()):
            color, heatmap, label, count = frames[arm_id]
            tile = self._make_arm_tile(color, heatmap, label, count,
                                       arm_id, summary, tw, th)
            if arm_id <= 1:
                top_row.append(tile)
            else:
                bottom_row.append(tile)

        # 状态面板
        status_panel = self._make_status_panel(summary, tw, th)

        # 补齐空白位
        while len(top_row) < 2:
            top_row.append(np.zeros((th, tw, 3), dtype=np.uint8))
        while len(bottom_row) < 2:
            bottom_row.append(np.zeros((th, tw, 3), dtype=np.uint8))

        # 将状态面板放在右下角
        if len(frames) < 4:
            bottom_row[-1] = status_panel
        else:
            bottom_row.append(status_panel)

        top = np.hstack(top_row[:2])
        bottom = np.hstack(bottom_row[:2])
        return np.vstack([top, bottom])

    def _build_vertical(self, frames, summary, tw, th) -> np.ndarray:
        """垂直布局：所有臂视图叠放于左侧 + 右侧状态栏。"""
        arm_tiles = []
        for arm_id in sorted(frames.keys()):
            color, heatmap, label, count = frames[arm_id]
            tile = self._make_arm_tile(color, heatmap, label, count,
                                       arm_id, summary, tw // 2, th // 2)
            arm_tiles.append(tile)

        status = self._make_status_panel(summary, tw // 2, th)
        left_col = np.vstack(arm_tiles) if arm_tiles else np.zeros(
            (th, tw // 2, 3), dtype=np.uint8
        )
        return np.hstack([left_col, status])

    def _make_arm_tile(self, color, heatmap, label, count,
                       arm_id, summary, tw, th) -> np.ndarray:
        """制作单臂显示块：彩色图（左）+ GGCNN 热力图（右）拼接。"""
        half_w = tw // 2
        h = th

        color_resized = cv2.resize(color, (half_w, h))
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (half_w, h))
            # 灰度热力图转三通道彩色
            if len(heatmap_resized.shape) == 2:
                heatmap_resized = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8),
                    cv2.COLORMAP_HOT
                )
        else:
            heatmap_resized = np.zeros((h, half_w, 3), dtype=np.uint8)

        tile = np.hstack([color_resized, heatmap_resized])

        # 臂名称 + 状态 + 抓取计数
        arm_name = self._config.get_arm(arm_id)
        name_str = arm_name.name if arm_name else f"Arm-{arm_id}"
        cv2.putText(tile, f"{name_str} | {label} | #{count}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, STATE_COLORS.get(label, (255, 255, 255)), 1)

        return tile

    def _make_status_panel(self, summary, tw, th) -> np.ndarray:
        """制作全局状态面板。"""
        panel = np.zeros((th, tw, 3), dtype=np.uint8)

        arm_states = summary.get('arm_states', {})
        grasp_counts = summary.get('grasp_counts', {})
        zones = summary.get('zones', {})
        any_hazard = summary.get('any_hazard', False)

        y = 30
        cv2.putText(panel, "=== SYSTEM STATUS ===", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25

        # HAZARD 闪烁警告
        if any_hazard:
            self._hazard_blink = not self._hazard_blink
            if self._hazard_blink:
                cv2.putText(panel, "!!! HAZARD !!!  Press 'r' to clear",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            y += 30

        # 各臂状态列表
        for arm_id in sorted(arm_states.keys()):
            state = arm_states[arm_id]
            count = grasp_counts.get(arm_id, 0)
            color = STATE_COLORS.get(state, (255, 255, 255))
            arm_name = self._config.get_arm(arm_id)
            name_str = arm_name.name if arm_name else f"Arm-{arm_id}"
            cv2.putText(panel, f"{name_str}: {state} (grasps:{count})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 20

        y += 10
        cv2.putText(panel, "--- Zone Ownership ---", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20

        # 区域占用状态
        for zone_name, info in sorted(zones.items()):
            state = info['state']
            holder = info.get('holder')
            if state == 'FREE':
                color = (100, 100, 100)
                text = f"  {zone_name}: FREE"
            elif state == 'HAZARD':
                color = (0, 0, 255)
                text = f"  {zone_name}: HAZARD!"
            else:
                color = (0, 200, 0)
                text = f"  {zone_name}: Arm-{holder}"
            cv2.putText(panel, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 18

        y += 10
        cv2.putText(panel, "Keys: q/ESC=Quit  r=Clear HAZARD  s=Summary",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1)
        y += 20

        # 臂间实时安全距离显示
        arm_positions = summary.get('arm_positions', {})
        if len(arm_positions) >= 2:
            y += 5
            cv2.putText(panel, "--- Inter-Arm Distance ---", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 18
            ids = sorted(arm_positions.keys())
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    pa = arm_positions[ids[i]]
                    pb = arm_positions[ids[j]]
                    if len(pa) >= 3 and len(pb) >= 3:
                        dist = interarm_distance(pa, pb)
                        # 颜色：绿(≥100) / 橙(50-100) / 红(<50)
                        color = ((0, 255, 0) if dist >= 100
                                 else (0, 165, 255) if dist >= 50
                                 else (0, 0, 255))
                        cv2.putText(
                            panel,
                            f"  Arm-{ids[i]} <-> Arm-{ids[j]}: {dist:.0f}mm",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                        )
                        y += 18

        return panel
