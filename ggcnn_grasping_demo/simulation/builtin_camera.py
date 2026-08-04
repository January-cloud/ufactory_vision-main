#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
builtin_camera — 本地合成摄像头
================================
生成模拟的 RGB-D 图像，无需任何硬件或外部服务器。提供与 RealSenseCamera
和 SimCamera 兼容的接口，可作为 drop-in 替换。

合成的场景包含:
  - 一个平坦桌面平面（可配置距离）
  - 可选的矩形/圆形物体（随机位置，每帧微动模拟真实相机抖动）
  - 高斯噪声模拟传感器噪声

使用方式:
    from builtin_camera import BuiltinCamera

    cam = BuiltinCamera(width=640, height=480)
    color, depth = cam.get_images()       # 与 RealSenseCamera 接口一致
    ci, di = cam.get_intrinsics()         # 获取内参
"""

import cv2
import math
import logging
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import namedtuple

logger = logging.getLogger(__name__)

# 模拟 pyrealsense2.intrinsics 对象的轻量 namedtuple
Intrinsics = namedtuple('Intrinsics', ['fx', 'fy', 'ppx', 'ppy', 'width', 'height'])


class BuiltinCamera:
    """本地合成摄像头 — 纯 Python 生成模拟 RGB-D 图像。

    提供与 RealSenseCamera 兼容的接口:
      - get_images(align=True) → (color_bgr, depth_float32)
      - get_intrinsics() → (color_intrin, depth_intrin)
      - stop() → 释放资源

    合成场景:
      - 桌面平面: 位于 table_depth_m 处
      - 物体: 桌面上的矩形块，高度可配置，每帧位置有微小随机抖动
      - 噪声: 高斯噪声 + 轻微模糊模拟真实传感器
    """

    # 默认内参（D435 典型值，可配置覆盖）
    DEFAULT_FX = 615.0
    DEFAULT_FY = 615.0

    # 默认物体配置（2-3 个物块，GGCNN 可检测到有意义的目标）
    DEFAULT_OBJECTS = [
        {"cx": 280, "cy": 200, "size_x": 55, "size_y": 50, "height_m": 0.030},
        {"cx": 380, "cy": 270, "size_x": 50, "size_y": 45, "height_m": 0.025},
        {"cx": 310, "cy": 250, "size_x": 45, "size_y": 40, "height_m": 0.028},
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
            objects:        物体列表，每项 dict:
                            {"cx": 像素, "cy": 像素, "size_x": 像素, "size_y": 像素, "height_m": 米}
                            None 则使用默认物体
            noise_std_m:    深度噪声标准差 (米)
            jitter_px:      每帧物体位置随机抖动范围 (像素)
            seed:           随机种子 (None = 不固定)
        """
        self.width = width
        self.height = height
        self.table_depth_m = table_depth_m
        self.noise_std_m = noise_std_m
        self.jitter_px = jitter_px

        self._fx = fx if fx is not None else self.DEFAULT_FX
        self._fy = fy if fy is not None else self.DEFAULT_FY
        cx = width / 2.0
        cy = height / 2.0

        # 构造 intrinsics 对象
        self._color_intrin = Intrinsics(
            fx=self._fx, fy=self._fy,
            ppx=cx, ppy=cy,
            width=width, height=height,
        )
        self._depth_intrin = Intrinsics(
            fx=self._fx, fy=self._fy,
            ppx=cx, ppy=cy,
            width=width, height=height,
        )

        # 物体配置
        self._objects = objects if objects is not None else self.DEFAULT_OBJECTS

        # 随机数生成器
        self._rng = np.random.RandomState(seed)
        self._frame_count = 0

        logger.info(
            "BuiltinCamera 初始化: %dx%d fx=%.1f fy=%.1f table_depth=%.2fm objects=%d",
            width, height, self._fx, self._fy, table_depth_m, len(self._objects)
        )

    # ── 公开接口（与 RealSenseCamera 兼容）──────────────────────────────────

    def get_images(self, align: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """生成一帧合成彩色图和深度图。

        参数:
            align: 忽略（合成图像自然已配准）

        返回:
            (color_bgr, depth_float32)
            color_bgr: uint8 ndarray (H, W, 3) BGR 格式
            depth_float32: float32 ndarray (H, W) 单位米，0 或 NaN 表示无效
        """
        depth = self._generate_depth()
        color = self._depth_to_color(depth)
        self._frame_count += 1
        return color, depth

    def get_intrinsics(self, align: bool = False):
        """获取相机内参。

        返回:
            (color_intrin, depth_intrin)
            每个都是包含 fx, fy, ppx, ppy, width, height 的对象
        """
        return self._color_intrin, self._depth_intrin

    def stop(self):
        """释放资源（无实际操作）。"""
        logger.info("BuiltinCamera 停止 (共 %d 帧)", self._frame_count)
        self._frame_count = 0

    # ── 内参更新 ────────────────────────────────────────────────────────────

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """手动设置相机内参。"""
        self._fx, self._fy = fx, fy
        self._color_intrin = Intrinsics(fx=fx, fy=fy, ppx=cx, ppy=cy,
                                        width=self.width, height=self.height)
        self._depth_intrin = Intrinsics(fx=fx, fy=fy, ppx=cx, ppy=cy,
                                        width=self.width, height=self.height)
        logger.info("内参已更新: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", fx, fy, cx, cy)

    # ── 场景物体管理 ──────────────────────────────────────────────────────

    def set_objects(self, objects: List[Dict]):
        """更新场景中的物体列表。"""
        self._objects = list(objects)
        logger.info("物体列表已更新: %d 个物体", len(self._objects))

    def add_object(self, cx: float, cy: float, size_x: float, size_y: float,
                   height_m: float):
        """向场景中添加一个物体。"""
        self._objects.append({
            "cx": cx, "cy": cy,
            "size_x": size_x, "size_y": size_y,
            "height_m": height_m,
        })

    def clear_objects(self):
        """清空所有物体（仅保留桌面平面）。"""
        self._objects.clear()

    # ── 图像生成（内部方法）────────────────────────────────────────────────

    def _generate_depth(self) -> np.ndarray:
        """生成一帧合成深度图 (float32, 米)。"""
        h, w = self.height, self.width

        # 基础桌面平面
        depth = np.full((h, w), self.table_depth_m, dtype=np.float32)

        # 绘制每个物体（在桌面上方的矩形块）
        for obj in self._objects:
            # 添加随机抖动模拟真实相机帧间变化
            jx = self._rng.uniform(-self.jitter_px, self.jitter_px)
            jy = self._rng.uniform(-self.jitter_px, self.jitter_px)

            cx = int(obj["cx"] + jx)
            cy = int(obj["cy"] + jy)
            sx = int(obj["size_x"])
            sy = int(obj["size_y"])
            obj_depth = self.table_depth_m - obj["height_m"]

            # 矩形物体区域
            x1 = max(0, cx - sx // 2)
            x2 = min(w, cx + sx // 2)
            y1 = max(0, cy - sy // 2)
            y2 = min(h, cy + sy // 2)

            if x2 > x1 and y2 > y1:
                depth[y1:y2, x1:x2] = obj_depth

        # 添加高斯噪声
        noise = self._rng.randn(h, w).astype(np.float32) * self.noise_std_m
        depth += noise

        # 轻微高斯模糊（模拟真实传感器的光学模糊 + 像素填充效应）
        depth = cv2.GaussianBlur(depth, (3, 3), 0.8)

        # 0 或负值替换为 NaN（模拟无效区域，如桌面边缘以外）
        depth[depth <= 0] = math.nan

        return depth.astype(np.float32)

    def _depth_to_color(self, depth: np.ndarray) -> np.ndarray:
        """将深度图转换为伪彩色图 (BGR uint8) 用于可视化。

        使用 JET 色映射并反转，使近处物体呈暖色 (红/黄)，
        远处桌面呈冷色 (蓝/青)，符合直觉。
        """
        # 归一化深度到 [0, 1]（忽略 NaN）
        valid = ~np.isnan(depth)
        if not np.any(valid):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        d_min = np.nanmin(depth)
        d_max = np.nanmax(depth)
        if d_max - d_min < 1e-6:
            normalized = np.zeros_like(depth)
        else:
            normalized = (depth - d_min) / (d_max - d_min)

        # NaN → 0
        normalized = np.nan_to_num(normalized, nan=0.0)

        # 反转：近处亮、远处暗 → 适合于 JET 映射（暖色=近, 冷色=远）
        inverted = (1.0 - normalized).astype(np.float32)

        # 转为 uint8
        display = (inverted * 255).clip(0, 255).astype(np.uint8)

        # JET 色映射
        color = cv2.applyColorMap(display, cv2.COLORMAP_JET)

        return color

    # ── 属性 ────────────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """已生成的帧数。"""
        return self._frame_count


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python builtin_camera.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    print("=" * 60)
    print("BuiltinCamera 自测")
    print("=" * 60)

    # 1. 基本初始化和内参
    print("\n[1] 初始化 + 内参测试...")
    cam = BuiltinCamera(width=640, height=480, seed=42)
    ci, di = cam.get_intrinsics()
    print(f"    Color: fx={ci.fx} fy={ci.fy} ppx={ci.ppx} ppy={ci.ppy} {ci.width}x{ci.height}")
    print(f"    Depth: fx={di.fx} fy={di.fy} ppx={di.ppx} ppy={di.ppy} {di.width}x{di.height}")
    assert ci.width == 640 and ci.height == 480, "Resolution mismatch"
    print("    [PASS] 内参")

    # 2. 单帧生成
    print("\n[2] 单帧合成图像测试...")
    color, depth = cam.get_images()
    print(f"    Color: {color.shape} {color.dtype} 值域 [{color.min()}, {color.max()}]")
    print(f"    Depth: {depth.shape} {depth.dtype} 值域 [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]")
    assert color.shape == (480, 640, 3), f"Color shape mismatch: {color.shape}"
    assert color.dtype == np.uint8, f"Color dtype mismatch: {color.dtype}"
    assert depth.shape == (480, 640), f"Depth shape mismatch: {depth.shape}"
    assert depth.dtype == np.float32, f"Depth dtype mismatch: {depth.dtype}"
    # 应有有效深度值
    valid_count = np.sum(~np.isnan(depth))
    print(f"    有效深度像素: {valid_count}/{depth.size} ({100*valid_count/depth.size:.1f}%)")
    assert valid_count > depth.size * 0.5, "Too many NaN pixels"
    print("    [PASS] 单帧生成")

    # 3. 多帧一致性（帧间有微小变化但不大）
    print("\n[3] 帧间变化测试...")
    _, d1 = cam.get_images()
    _, d2 = cam.get_images()
    diff = np.nanmean(np.abs(d1 - d2))
    print(f"    帧间平均深度差: {diff:.6f}m")
    # 应有微小差异（噪声 + 抖动），但不能太大
    assert diff > 0, "Frames should differ due to noise"
    assert diff < 0.1, f"Frame difference too large: {diff}"
    print("    [PASS] 帧间变化")

    # 4. 多帧后 frame_count 正确
    print("\n[4] 帧计数测试...")
    for _ in range(5):
        cam.get_images()
    print(f"    帧计数: {cam.frame_count}")
    assert cam.frame_count == 8, f"Frame count should be 8, got {cam.frame_count}"  # 3 + 5
    print("    [PASS] 帧计数")

    # 5. 无物体场景
    print("\n[5] 无物体场景 (仅桌面平面)...")
    cam2 = BuiltinCamera(width=320, height=240, objects=[], seed=1)
    _, d3 = cam2.get_images()
    # 所有值应接近 table_depth_m
    mean_d = np.nanmean(d3)
    print(f"    平均深度: {mean_d:.4f}m (期望 ~0.50m)")
    assert abs(mean_d - 0.50) < 0.01, f"Mean depth deviates too much: {mean_d}"
    print("    [PASS] 无物体场景")

    # 6. set_intrinsics 测试
    print("\n[6] 内参更新测试...")
    cam.set_intrinsics(700.0, 700.0, 300.0, 220.0)
    ci2, _ = cam.get_intrinsics()
    print(f"    fx={ci2.fx} fy={ci2.fy} cx={ci2.ppx} cy={ci2.ppy}")
    assert ci2.fx == 700.0 and ci2.fy == 700.0
    assert ci2.ppx == 300.0 and ci2.ppy == 220.0
    print("    [PASS] 内参更新")

    # 7. add_object / clear_objects 测试
    print("\n[7] 物体管理测试...")
    cam3 = BuiltinCamera(width=320, height=240, objects=[], seed=2)
    _, d_empty = cam3.get_images()
    cam3.add_object(160, 120, 60, 60, 0.04)
    _, d_with_obj = cam3.get_images()
    # 有物体的帧平均深度应更小（物体比桌面更近）
    print(f"    空场景平均深度: {np.nanmean(d_empty):.4f}m")
    print(f"    有物体平均深度: {np.nanmean(d_with_obj):.4f}m")
    assert np.nanmean(d_with_obj) < np.nanmean(d_empty), "Object should make avg depth closer"
    cam3.clear_objects()
    _, d_cleared = cam3.get_images()
    assert abs(np.nanmean(d_cleared) - np.nanmean(d_empty)) < 0.01
    print("    [PASS] 物体管理")

    cam.stop()
    cam2.stop()
    cam3.stop()
    print(f"\n自测完成 [PASS]")
