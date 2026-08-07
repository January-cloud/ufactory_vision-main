#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_visualizer — 三臂仿真组合显示窗口
=======================================
将三台机械臂的相机画面 + GGCNN 热力图 + 全局摄像头 + 系统状态信息
组合到单个 OpenCV 窗口，独立渲染线程，每秒约 30 帧。

布局 (2×2):
┌─────────────────────┬─────────────────────┐
│  Arm-0 彩色图+热力图 │  Arm-1 彩色图+热力图 │
├─────────────────────┼─────────────────────┤
│  Arm-2 彩色图+热力图 │  全局摄像头 + 状态面板 │
└─────────────────────┴─────────────────────┘

按键:
  q / ESC   退出
  r         清除 HAZARD 区域 (需人工确认安全)
  s         打印系统摘要
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import cv2
from typing import Dict, Optional

# ── 包内路径引导 ──
_sim_3arm_dir = os.path.dirname(os.path.abspath(__file__))
_simulation_dir = os.path.dirname(_sim_3arm_dir)
_demo_dir = os.path.dirname(_simulation_dir)
for _p in (_sim_3arm_dir, _simulation_dir, _demo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multi_arm.config import SystemConfig
from multi_arm.collision_avoidance import interarm_distance
from simulation.global_camera import GlobalCamera
from simulation.sim_3arm.sim_coordinator import SimCoordinator


logger = logging.getLogger(__name__)

# 状态对应的显示颜色 (BGR) — 与真机 visualizer 保持一致
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


class SimVisualizer:
    """三臂仿真组合可视化。

    在独立线程中运行，从各 SimArmController 拉取帧数据、从 GlobalCamera
    拉取全局画面、从 SimCoordinator 拉取状态，拼合成一个 2×2 窗口。
    """

    def __init__(self, config: SystemConfig,
                 coordinator: SimCoordinator,
                 global_camera: GlobalCamera):
        """
        参数:
            config:        SystemConfig (真机配置，含 win_name/arms)
            coordinator:   SimCoordinator
            global_camera: 全局俯瞰摄像头实例
        """
        self._config = config
        self._coord = coordinator
        self._global_camera = global_camera

        # arm_id → SimArmController 引用 (用于拉取帧)
        self._controllers: Dict[int, object] = {}

        self._window_name = config.win_name
        self._render_thread: Optional[threading.Thread] = None
        self._running = False

        # HAZARD 闪烁状态
        self._hazard_blink = False
        self._hazard_last_toggle = 0.0

    def attach_controller(self, ctrl):
        """绑定 SimArmController 实例 (按 arm_id)。"""
        self._controllers[ctrl._cfg.arm_id] = ctrl

    def start(self):
        """启动渲染线程。"""
        if self._running:
            return
        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="SimVisualizer",
            daemon=True,
        )
        self._render_thread.start()
        logger.info("SimVisualizer 已启动")

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
        """渲染主循环 (约 30Hz)。

        注意：cv2.namedWindow / imshow / waitKey / destroyWindow 必须全部
        在同一个线程中调用（Windows 下跨线程访问高GUI窗口会阻塞挂起），
        因此窗口创建也放在此渲染线程内。
        """
        logger.info("SimVisualizer 渲染循环开始")
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 1280, 720)

        while self._running and not self._coord.global_stop_event.is_set():
            combined = self._build_combined_frame()
            if combined is not None:
                cv2.imshow(self._window_name, combined)

            key = cv2.waitKey(33) & 0xFF
            if key == ord('q') or key == 27:
                self._coord.broadcast_stop()
                break
            elif key == ord('r'):
                if self._coord.any_hazard():
                    self._coord.clear_hazard()
                    logger.info("[SimVisualizer] HAZARD 已人工清除")
            elif key == ord('s'):
                summary = self._coord.get_summary()
                logger.info(f"[SimVisualizer] 系统摘要: {summary}")

        # 与窗口创建同线程销毁，避免阻塞
        cv2.destroyWindow(self._window_name)
        logger.info("SimVisualizer 停止")

    # ═══════════════════════════════════════════════════════════════════
    # 帧构建 (2×2 布局)
    # ═══════════════════════════════════════════════════════════════════

    def _build_combined_frame(self) -> Optional[np.ndarray]:
        """拼合三臂画面 + 全局摄像头 + 状态面板。"""
        arm_count = len(self._config.arms)
        if arm_count == 0:
            return None

        # 收集所有臂的最新帧
        frames = {}
        for arm_id, ctrl in sorted(self._controllers.items()):
            color, heatmap, label, count = ctrl.get_latest_frame()
            if color is not None:
                frames[arm_id] = (color, heatmap, label, count)

        summary = self._coord.get_summary()

        tw, th = 640, 360  # 每格尺寸 (2×2 布局)

        # 全局摄像头画面
        try:
            g_color, _ = self._global_camera.get_images(align=True)
        except Exception:
            g_color = np.zeros((th, tw, 3), dtype=np.uint8)
            cv2.putText(g_color, "GLOBAL CAMERA ERROR", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 依次构建 4 个格子
        tiles = {}
        for arm_id in sorted(self._config.arms, key=lambda a: a.arm_id):
            arm_cfg = self._config.get_arm(arm_id.arm_id)
            f = frames.get(arm_id.arm_id)
            if f is None:
                # 等待画面
                tile = np.zeros((th, tw, 3), dtype=np.uint8)
                cv2.putText(tile, f"Arm-{arm_id.arm_id}: waiting...",
                            (20, th//2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)
            else:
                color, heatmap, label, count = f
                tile = self._make_arm_tile(
                    color, heatmap, label, count,
                    arm_cfg.name if arm_cfg else f"Arm-{arm_id.arm_id}",
                    summary, tw, th
                )
            tiles[arm_id.arm_id] = tile

        # 全局摄像头 + 状态面板
        global_tile = self._make_global_tile(
            g_color, summary, tw, th
        )
        tiles[3] = global_tile

        # 2×2 拼合: [0,1; 2,3]
        top = np.hstack([tiles.get(0, np.zeros((th, tw, 3), np.uint8)),
                         tiles.get(1, np.zeros((th, tw, 3), np.uint8))])
        bottom = np.hstack([tiles.get(2, np.zeros((th, tw, 3), np.uint8)),
                            tiles.get(3, np.zeros((th, tw, 3), np.uint8))])
        return np.vstack([top, bottom])

    # ═══════════════════════════════════════════════════════════════════
    # 格子弹窗
    # ═══════════════════════════════════════════════════════════════════

    def _make_arm_tile(self, color, heatmap, label, count,
                       arm_name, summary, tw, th) -> np.ndarray:
        """制作单臂显示块：彩色图（左）+ GGCNN 热力图（右）+ 状态文字。"""
        half_w = tw // 2
        h = th

        color_resized = cv2.resize(color, (half_w, h))
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (half_w, h))
            if len(heatmap_resized.shape) == 2:
                heatmap_resized = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8),
                    cv2.COLORMAP_HOT
                )
        else:
            heatmap_resized = np.zeros((h, half_w, 3), dtype=np.uint8)

        tile = np.hstack([color_resized, heatmap_resized])

        cv2.putText(tile, f"{arm_name} | {label} | #{count}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, STATE_COLORS.get(label, (255, 255, 255)), 1)
        return tile

    def _make_global_tile(self, g_color, summary, tw, th) -> np.ndarray:
        """制作全局摄像头 + 状态面板：全局画面上叠加系统状态。"""
        tile = cv2.resize(g_color, (tw, th))

        # 半透明黑色覆盖带，用于承载状态文字
        overlay = tile.copy()
        cv2.rectangle(overlay, (0, 0), (tw, 100), (0, 0, 0), -1)
        tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)

        # 标题
        cv2.putText(tile, "=== GLOBAL VIEW / SYSTEM STATUS ===",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        arm_states = summary.get('arm_states', {})
        grasp_counts = summary.get('grasp_counts', {})
        any_hazard = summary.get('any_hazard', False)

        y = 38
        if any_hazard:
            self._hazard_blink = not self._hazard_blink
            if self._hazard_blink:
                cv2.putText(tile, "!!! HAZARD !!! Press 'r' to clear",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255), 2)
            y += 22

        for arm_id in sorted(arm_states.keys()):
            state = arm_states[arm_id]
            cnt = grasp_counts.get(arm_id, 0)
            color = STATE_COLORS.get(state, (255, 255, 255))
            arm_cfg = self._config.get_arm(arm_id)
            name_str = arm_cfg.name if arm_cfg else f"Arm-{arm_id}"
            cv2.putText(tile, f"{name_str}: {state} (grasps:{cnt})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 18

        # 臂间距离
        arm_positions = summary.get('arm_positions', {})
        ids = sorted(arm_positions.keys())
        if len(ids) >= 2:
            y += 4
            cv2.putText(tile, "--- Distances ---",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1)
            y += 16
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    pa, pb = arm_positions[ids[i]], arm_positions[ids[j]]
                    dist = interarm_distance(pa, pb)
                    dcolor = ((0, 255, 0) if dist >= 100
                              else (0, 165, 255) if dist >= 50
                              else (0, 0, 255))
                    cv2.putText(tile,
                                f"  {ids[i]}-{ids[j]}: {dist:.0f}mm",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, dcolor, 1)
                    y += 16

        cv2.putText(tile, "Keys: q=Quit r=ClearHazard s=Summary",
                    (10, th - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1)

        return tile


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python sim_visualizer.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("SimVisualizer 自测")
    print("=" * 60)

    from simulation.sim_3arm.sim_coordinator import load_sim_config
    from simulation.sim_3arm.sim_arm_controller import SimArmController
    from simulation.simulation_client import SimulationClient

    cfg_path = os.path.join(_simulation_dir, 'config_sim_3arms.json')
    sim_cfg = load_sim_config(cfg_path)
    client = SimulationClient("http://127.0.0.1:1", timeout=1.0, retries=0)
    coord = SimCoordinator(sim_cfg, client=client)

    gc_cfg = sim_cfg.global_camera
    global_cam = GlobalCamera(width=gc_cfg.width, height=gc_cfg.height,
                              objects=gc_cfg.objects)

    viz = SimVisualizer(sim_cfg.system_config, coord, global_cam)

    # 创建 3 个控制器 (无相机模式，快速)
    controllers = []
    for arm_cfg in sim_cfg.arms:
        ctrl = SimArmController(sim_cfg, arm_cfg, coord,
                                no_camera=True)
        controllers.append(ctrl)
        viz.attach_controller(ctrl)

    # 运行 3 秒
    print("\n[1] 启动 3 臂 + 可视化, 运行 3 秒...")
    for ctrl in controllers:
        ctrl.start()
    viz.start()
    time.sleep(3.0)

    # 验证各臂发布帧
    print("\n[2] 验证帧数据...")
    for ctrl in controllers:
        color, heatmap, label, count = ctrl.get_latest_frame()
        assert color is not None, f"Arm-{ctrl._cfg.arm_id} 无帧"
        print(f"    Arm-{ctrl._cfg.arm_id}: {color.shape} label={label} "
              f"count={count}")
    print("    [PASS]")

    print("\n[3] 停止...")
    viz.stop()
    coord.broadcast_stop()
    for ctrl in controllers:
        ctrl.stop()
    for ctrl in controllers:
        ctrl.join(timeout=3.0)
    viz.join(timeout=3.0)

    coord.close()
    print("\n自测完成 [PASS]")
