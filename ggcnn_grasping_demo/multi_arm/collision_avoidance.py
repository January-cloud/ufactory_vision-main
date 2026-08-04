#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collision_avoidance — 区域划分与碰撞检测工具
=============================================
提供:
  - 为 N 个并排机械臂自动计算独占区和协调区边界
  - 点是否在区域内 / 区域是否重叠 / 点到区域最短距离
  - 两臂末端欧氏距离计算
  - 轨迹是否穿越危险区域的预判
"""

import numpy as np
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════
# 区域基础数学
# ═══════════════════════════════════════════════════════════════════════════

def point_in_zone(x: float, y: float, zone: List[float]) -> bool:
    """判断点 (x, y) 是否在矩形区域 [x_min, x_max, y_min, y_max] 内。"""
    return zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3]


def zones_overlap(a: List[float], b: List[float]) -> bool:
    """判断两个矩形区域是否有重叠。"""
    return (a[0] < b[1] and a[1] > b[0] and
            a[2] < b[3] and a[3] > b[2])


def min_distance_to_zone(x: float, y: float, zone: List[float]) -> float:
    """计算点 (x, y) 到矩形区域的最短欧氏距离（点在区域内返回 0）。"""
    dx = max(zone[0] - x, 0, x - zone[1])
    dy = max(zone[2] - y, 0, y - zone[3])
    return np.sqrt(dx * dx + dy * dy)


def interarm_distance(pos_a: List[float], pos_b: List[float]) -> float:
    """计算两个机械臂末端 (EE) 之间的三维欧氏距离 (mm)。"""
    a = np.array(pos_a[:3], dtype=float)
    b = np.array(pos_b[:3], dtype=float)
    return float(np.linalg.norm(a - b))


# ═══════════════════════════════════════════════════════════════════════════
# 区域自动划分
# ═══════════════════════════════════════════════════════════════════════════

def compute_zones_linear(
    arm_count: int,
    base_positions: List[Tuple[float, float]],
    table_x_range: Tuple[float, float],
    zone_overlap_margin: float = 50.0,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    为线形并排布局自动计算独占区和协调区。

    参数:
        arm_count:           机械臂数量
        base_positions:      每个臂基座的 (x_mm, y_mm)，按 Y 坐标排序
        table_x_range:       工作台 X 范围 (x_min, x_max)
        zone_overlap_margin: 协调区宽度 (mm)，相邻独占区各向外延伸此宽度

    返回:
        exclusive_zones:      每臂独占区 [x_min, x_max, y_min, y_max]
        coordination_zones:   相邻臂之间的协调区 [(zone, (arm_a, arm_b)), ...]

    假设:
        - 臂沿 Y 轴排列（从左到右 Y 递增）
        - 每臂独占区 X 范围 = table_x_range
        - 独占区 Y 分界线 = 相邻臂基座 Y 位置的中点
    """
    if arm_count < 1:
        return [], []

    # 按 Y 排序
    sorted_indices = sorted(range(arm_count),
                            key=lambda i: base_positions[i][1])
    sorted_bases = [base_positions[i] for i in sorted_indices]

    # 计算 Y 分界线（臂基座之间的中点）
    y_boundaries = []
    for i in range(arm_count - 1):
        mid_y = (sorted_bases[i][1] + sorted_bases[i + 1][1]) / 2.0
        y_boundaries.append(mid_y)

    # 独占区：相邻分界线之间，首尾臂向两侧无限延伸
    exclusive_zones_raw = []
    for i in range(arm_count):
        y_min = y_boundaries[i - 1] if i > 0 else table_x_range[0] - 1000
        y_max = y_boundaries[i] if i < arm_count - 1 else table_x_range[0] + 1000
        zone = [table_x_range[0], table_x_range[1], y_min, y_max]
        exclusive_zones_raw.append(zone)

    # 协调区 = 相邻独占区之间的 overlap_margin 宽度条带
    coordination_zones = []
    for i in range(arm_count - 1):
        a_idx = sorted_indices[i]
        b_idx = sorted_indices[i + 1]
        mid = y_boundaries[i]
        coord_zone = [
            table_x_range[0],
            table_x_range[1],
            mid - zone_overlap_margin,
            mid + zone_overlap_margin,
        ]
        coordination_zones.append((coord_zone, (a_idx, b_idx)))

    # 按原始 arm_id 顺序返回独占区
    exclusive_zones = [None] * arm_count
    for i, idx in enumerate(sorted_indices):
        exclusive_zones[idx] = exclusive_zones_raw[i]

    return exclusive_zones, coordination_zones


# ═══════════════════════════════════════════════════════════════════════════
# 轨迹碰撞预判
# ═══════════════════════════════════════════════════════════════════════════

def trajectory_crosses_zone(
    start_xyz: Tuple[float, float, float],
    end_xyz: Tuple[float, float, float],
    zone: List[float],
    safety_z: float = 350.0,
) -> bool:
    """
    判断从 start 到 end 的直线轨迹是否经过 zone 的危险区域（Z < safety_z）。

    仅在 XY 投影穿过 zone 且路径上有 Z 低于 safety_z 的段时返回 True。
    如果全程在 safety_z 以上（高位过顶），不视为危险。
    """
    # 如果全程在高位，安全
    if start_xyz[2] >= safety_z and end_xyz[2] >= safety_z:
        return False

    # 提取 XY 分量
    sx, sy = start_xyz[0], start_xyz[1]
    ex, ey = end_xyz[0], end_xyz[1]

    # 检查线段端点是否在区域内
    if point_in_zone(sx, sy, zone) or point_in_zone(ex, ey, zone):
        return True

    # 检查线段是否与矩形四边相交
    rect_edges = [
        (zone[0], zone[2], zone[1], zone[2]),  # 下边
        (zone[1], zone[2], zone[1], zone[3]),  # 右边
        (zone[1], zone[3], zone[0], zone[3]),  # 上边
        (zone[0], zone[3], zone[0], zone[2]),  # 左边
    ]
    for x1, y1, x2, y2 in rect_edges:
        if _segments_intersect(sx, sy, ex, ey, x1, y1, x2, y2):
            return True
    return False


def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4) -> bool:
    """二维线段相交检测（跨立实验）。"""
    def _cross(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    d1 = _cross(x3, y3, x4, y4, x1, y1)
    d2 = _cross(x3, y3, x4, y4, x2, y2)
    d3 = _cross(x1, y1, x2, y2, x3, y3)
    d4 = _cross(x1, y1, x2, y2, x4, y4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # 共线情况
    if d1 == 0 and _on_segment(x3, y3, x4, y4, x1, y1): return True
    if d2 == 0 and _on_segment(x3, y3, x4, y4, x2, y2): return True
    if d3 == 0 and _on_segment(x1, y1, x2, y2, x3, y3): return True
    if d4 == 0 and _on_segment(x1, y1, x2, y2, x4, y4): return True
    return False


def _on_segment(x1, y1, x2, y2, px, py) -> bool:
    """判断点 (px, py) 是否在线段上。"""
    return (min(x1, x2) <= px <= max(x1, x2) and
            min(y1, y2) <= py <= max(y1, y2))


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== Zone Partition Test (3-arm linear) ===")
    bases = [(200, -250), (200, 0), (200, 250)]
    excl, coord = compute_zones_linear(3, bases, (100, 500), 50.0)
    for i, z in enumerate(excl):
        print(f"  Arm-{i} Exclusive: X[{z[0]:.0f},{z[1]:.0f}] Y[{z[2]:.0f},{z[3]:.0f}]")
    for z, (a, b) in coord:
        print(f"  Coord Arm-{a}<->Arm-{b}: X[{z[0]:.0f},{z[1]:.0f}] Y[{z[2]:.0f},{z[3]:.0f}]")

    print("\n=== Point/Zone Tests ===")
    print(f"  (150,-200) in Zone-0: {point_in_zone(150, -200, excl[0])}")
    print(f"  (150,0) in Zone-0: {point_in_zone(150, 0, excl[0])}")
    print(f"  Dist (400,200)->Zone-2: {min_distance_to_zone(400, 200, excl[2]):.1f}mm")
    print(f"  Arm0<->Arm1 EE dist: {interarm_distance([200,-200,380], [200,0,300]):.1f}mm")

    print("\n=== Trajectory Collision Tests ===")
    zone1 = excl[1]  # Arm-1 独占区
    # 高位过顶（安全）
    print(f"  High flyover: {trajectory_crosses_zone((150,-300,400),(150,300,400),zone1)} (expect False)")
    # 低位穿行（危险）
    print(f"  Low pass: {trajectory_crosses_zone((150,-300,100),(150,300,100),zone1)} (expect True)")
