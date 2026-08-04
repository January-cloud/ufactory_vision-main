#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成可视化图像: 合成深度图 + GGCNN 热力图，保存为 PNG 文件。"""

import os
import sys
import time
import math
import numpy as np
import cv2

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.join(_current_dir, '..')
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from grasp.ggcnn_torch import TorchGGCNN
from queue import Queue

# ── 参数 ──
OUT_DIR = _current_dir
W, H = 640, 480

# ── 生成合成深度图 (含彩色可视化) ──

def make_synthetic_depth(width=640, height=480, table_depth_m=0.5,
                         objects=None, noise_std_m=0.001, seed=42):
    """生成模拟桌面+物块的合成深度图。"""
    rng = np.random.RandomState(seed)
    depth = np.full((height, width), table_depth_m, dtype=np.float32)

    if objects is None:
        objects = [
            (290, 210, 55, 0.030),  # (cx, cy, size, height_m)
            (370, 270, 50, 0.028),
        ]

    for cx, cy, size, obj_h in objects:
        half = size // 2
        y1 = max(0, cy - half)
        y2 = min(height, cy + half)
        x1 = max(0, cx - half)
        x2 = min(width, cx + half)
        depth[y1:y2, x1:x2] = table_depth_m - obj_h

    noise = rng.randn(height, width).astype(np.float32) * noise_std_m
    depth += noise
    depth = cv2.GaussianBlur(depth, (5, 5), 1.0)
    return depth


def depth_to_colormap(depth_m, vmin=None, vmax=None):
    """深度图转伪彩色图像 (JET colormap)。"""
    if vmin is None:
        vmin = np.nanpercentile(depth_m, 1)
    if vmax is None:
        vmax = np.nanpercentile(depth_m, 99)
    normalized = np.clip((depth_m - vmin) / (vmax - vmin + 1e-8), 0, 1)
    normalized = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    # 深度越小 (越靠近相机, 即物体) = 偏红/黄; 深度越大 (桌面) = 偏蓝
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)
    return colored


# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  生成可视化图像演示")
print("=" * 60)

# 1. 生成深度图
print("\n[1] 生成合成深度图 (桌面 0.5m + 两个物块)...")
depth = make_synthetic_depth(W, H, table_depth_m=0.5, seed=42)
depth_colored = depth_to_colormap(depth)

# 在深度图上标注物块位置
depth_labeled = depth_colored.copy()
cv2.putText(depth_labeled, "SYNTHETIC DEPTH (table + 2 objects)", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.putText(depth_labeled, "Closer (warmer) = Object    Farther (cooler) = Table",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
cv2.putText(depth_labeled, f"Range: {depth.min():.3f} ~ {depth.max():.3f} m",
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

cv2.imwrite(os.path.join(OUT_DIR, "demo_input_depth.png"), depth_labeled)
print("  -> demo_input_depth.png")

# 2. GGCNN 推理
print("\n[2] 加载 GGCNN2 模型并推理...")
MODEL_FILE = os.path.join(_parent_dir, 'models', 'epoch_50_cornell')
K = np.array([[615.0, 0, W/2], [0, 615.0, H/2], [0, 0, 1]])

ggcnn = TorchGGCNN({
    'MODEL_FILE': MODEL_FILE,
    'OPEN_LOOP_HEIGHT': 0,
    'GGCNN_IN_THREAD': False,
    'DEPTH_CAM_K': K,
}, Queue(1), Queue(1))
time.sleep(0.5)

grasp_img, result = ggcnn.get_grasp_img(depth, K, 0.38)  # robot_z=0.38m (观察位)

# 3. 合成最终输出图 (左: 深度彩色图, 右: GGCNN 热力图)
print("\n[3] 合成对比图...")
# GGCNN 输出热力图的实际尺寸
gh, gw = grasp_img.shape[:2]
depth_resized = cv2.resize(depth_labeled, (gw, gh))
grasp_display = grasp_img.copy()

# 在热力图上标注信息
cv2.putText(grasp_display, "GGCNN2 HEATMAP (quality + grasp point)",
            (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
cv2.putText(grasp_display, "Green dot = best grasp pixel",
            (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

if result:
    cv2.putText(grasp_display,
                f"Grasp: x={result[0]:.3f} y={result[1]:.3f} z={result[2]:.3f}m ang={math.degrees(result[3]):.0f}deg",
                (5, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(grasp_display,
                f"Width: {result[4]:.0f}mm  DepthCenter: {result[5]:.0f}mm",
                (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

# 水平拼接
combined = np.hstack([depth_resized, grasp_display])

# 顶部标题栏
title_bar = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
cv2.putText(title_bar, "LEFT: Input Depth Map (JET colormap)",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(title_bar, "RIGHT: GGCNN2 Grasp Quality Heatmap (green = best grasp)",
            (depth_resized.shape[1] + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

combined = np.vstack([title_bar, combined])

cv2.imwrite(os.path.join(OUT_DIR, "demo_ggcnn_output.png"), combined)
print("  -> demo_ggcnn_output.png")

# 4. 输出数值结果
print(f"\n[4] GGCNN 推理结果:")
if result:
    print(f"    相机系抓取点: x={result[0]:.4f} y={result[1]:.4f} z={result[2]:.4f} m")
    print(f"    抓取角度:     {math.degrees(result[3]):.1f} deg")
    print(f"    推荐夹爪宽度: {result[4]:.1f} mm")
    print(f"    中心深度:     {result[5]:.1f} mm")
else:
    print(f"    未检测到有效抓取点")

print(f"\n完成! 查看以下文件:")
print(f"  {os.path.join(OUT_DIR, 'demo_input_depth.png')}")
print(f"  {os.path.join(OUT_DIR, 'demo_ggcnn_output.png')}")
