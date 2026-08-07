#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_camera — 全局俯瞰摄像头
================================
模拟安装在所有机械臂工作区上方的全局（俯视）摄像头，提供对整个场景的
鸟瞰视图。接口与 BuiltinCamera / RealSenseCamera 完全兼容。

与每臂的 "眼在手" 摄像头不同，全局摄像头固定在场景上方，视野覆盖
全部三臂的独占区与协调区，主要用于整体场景监控与可视化。

使用方式:
    from global_camera import GlobalCamera

    cam = GlobalCamera(width=640, height=480)
    color, depth = cam.get_images()       # 与 RealSenseCamera 接口一致
    ci, di = cam.get_intrinsics()
"""

import cv2
import math
import logging
from typing import List, Dict, Optional, Tuple

# 兼容两种导入方式:
#   直接运行: python global_camera.py      → from builtin_camera import ...
#   包内导入: from simulation import GlobalCamera → from .builtin_camera import ...
try:
    from builtin_camera import BuiltinCamera, Intrinsics
except ImportError:
    from .builtin_camera import BuiltinCamera, Intrinsics

logger = logging.getLogger(__name__)


class GlobalCamera(BuiltinCamera):
    """全局俯瞰摄像头 — 覆盖整个三臂工作区的合成鸟瞰视图。

    继承 BuiltinCamera，默认场景包含分布在画面左 / 中 / 右三列的物块，
    对应三台机械臂的独立工作区，便于操作人员总览整体抓取进程。
    """

    # 默认全局场景：物块分布在左 / 中 / 右三个区域
    DEFAULT_GLOBAL_OBJECTS = [
        # 左区 (对应 Arm-0 / Arm-Left)
        {"cx": 180, "cy": 220, "size_x": 50, "size_y": 45, "height_m": 0.030},
        {"cx": 160, "cy": 300, "size_x": 45, "size_y": 40, "height_m": 0.025},
        # 中区 (对应 Arm-1 / Arm-Center)
        {"cx": 320, "cy": 240, "size_x": 55, "size_y": 50, "height_m": 0.030},
        {"cx": 340, "cy": 320, "size_x": 45, "size_y": 40, "height_m": 0.028},
        # 右区 (对应 Arm-2 / Arm-Right)
        {"cx": 480, "cy": 220, "size_x": 50, "size_y": 45, "height_m": 0.030},
        {"cx": 500, "cy": 300, "size_x": 45, "size_y": 40, "height_m": 0.025},
    ]

    def __init__(self, width: int = 640, height: int = 480,
                 fx: float = None, fy: float = None,
                 table_depth_m: float = 0.50,
                 objects: Optional[List[Dict]] = None,
                 noise_std_m: float = 0.0015,
                 jitter_px: float = 3.0,
                 seed: int = None):
        """
        参数:
            width, height:  图像分辨率 (像素)
            fx, fy:         相机焦距 (像素)，None 则使用默认值
            table_depth_m:  桌面深度 (米)
            objects:        全局场景物块列表 (像素坐标)，None 则用默认全局场景
            noise_std_m:    深度噪声标准差 (米)
            jitter_px:      每帧物体位置随机抖动范围 (像素)
            seed:           随机种子
        """
        objects = objects if objects is not None else self.DEFAULT_GLOBAL_OBJECTS
        super().__init__(
            width=width, height=height,
            fx=fx, fy=fy,
            table_depth_m=table_depth_m,
            objects=objects,
            noise_std_m=noise_std_m,
            jitter_px=jitter_px,
            seed=seed,
        )
        logger.info(
            "GlobalCamera 初始化: %dx%d 全局场景物块 %d 个",
            width, height, len(self._objects)
        )

    # ── 可视化辅助 ────────────────────────────────────────────────────────

    def annotate_zones(self, color: Optional[dict] = None) -> Optional[dict]:
        """（占位方法）为后续版本预留：在全局画面上标注工作区边界。

        当前版本全局画面直接由 get_images() 提供，zone 标注由
        SimVisualizer 负责绘制，这里保留扩展点。
        """
        return color

    def __repr__(self):
        return (f"GlobalCamera({self.width}x{self.height}, "
                f"objects={len(self._objects)})")


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python global_camera.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import numpy as np
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("GlobalCamera 自测")
    print("=" * 60)

    cam = GlobalCamera(width=640, height=480, seed=42)

    # 1. 内参
    print("\n[1] 内参...")
    ci, di = cam.get_intrinsics()
    print(f"    fx={ci.fx} fy={ci.fy} cx={ci.ppx} cy={ci.ppy} {ci.width}x{ci.height}")
    assert ci.width == 640 and ci.height == 480
    print("    [PASS]")

    # 2. 单帧
    print("\n[2] 单帧生成...")
    color, depth = cam.get_images()
    print(f"    Color: {color.shape} {color.dtype}")
    print(f"    Depth: {depth.shape} {depth.dtype}")
    valid_count = int((~np.isnan(depth)).sum())
    print(f"    有效深度: {valid_count}/{depth.size} ({100*valid_count/depth.size:.1f}%)")
    assert color.shape == (480, 640, 3)
    assert depth.shape == (480, 640)
    assert valid_count > depth.size * 0.5
    print("    [PASS]")

    # 3. 帧间变化 (噪声+抖动)
    print("\n[3] 帧间变化...")
    _, d1 = cam.get_images()
    _, d2 = cam.get_images()
    diff = np.nanmean(np.abs(d1 - d2))
    print(f"    帧间平均深度差: {diff:.6f}m")
    assert 0 < diff < 0.1
    print("    [PASS]")

    # 4. 物块数量
    print("\n[4] 全局场景物块数量...")
    print(f"    objects = {len(cam._objects)} 个 (左/中/右三区)")
    assert len(cam._objects) >= 3
    print("    [PASS]")

    cam.stop()
    print(f"\n自测完成 [PASS]")
