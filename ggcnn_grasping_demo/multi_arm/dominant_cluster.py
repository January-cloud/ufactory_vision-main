#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dominant_cluster — 候选抓取点网格聚类
======================================
从候选缓冲区中找出最稳定的物块（出现频率最高的 XY 网格）。

原 run_rs_d435_grasp_lite6_new_best.py 第272行引用了此函数但未定义，
此处补全实现。逻辑与 detect_target_robust 内部聚类一致：
  - 按 XY 量化到 bin_mm 网格
  - 找命中次数最多的格子
  - 合并该格子及相邻 ±1 格内的所有点
  - 返回簇质心和命中数
"""

import numpy as np
from collections import Counter


def dominant_cluster(candidates, bin_mm):
    """
    从抓取候选列表中找出最密集的 XY 簇。

    参数:
        candidates: 候选目标列表，每项为 [x, y, z, roll, pitch, yaw]
                    基坐标系下的抓取目标 (mm, 度)
        bin_mm:     网格边长 (mm)，应小于物块间距、大于单物块抖动范围

    返回:
        (goal, count) 或 (None, 0)
        goal:  [x, y, z, 180, 0, yaw] 簇质心
        count: 簇内帧数
    """
    if not candidates or len(candidates) < 2:
        if len(candidates) == 1:
            return list(candidates[0]), 1
        return None, 0

    # 转为 NumPy 数组，shape (N, 6)
    a = np.array(candidates)

    # XY 量化到离散格子
    keys = np.round(a[:, :2] / bin_mm).astype(int)
    keys_t = [tuple(k) for k in keys]

    # 出现次数最多的格子
    best_key, best_cnt = Counter(keys_t).most_common(1)[0]

    # 取最优格子 ±1 邻域内的所有点（扩大簇，避免边界分裂）
    sel = [i for i, k in enumerate(keys_t)
           if abs(k[0] - best_key[0]) <= 1 and abs(k[1] - best_key[1]) <= 1]
    cluster = a[sel]

    # 对簇内点取均值作为最终目标
    g = np.mean(cluster, axis=0)
    return [float(g[0]), float(g[1]), float(g[2]), 180.0, 0.0, float(g[5])], len(cluster)


# ---------------------------------------------------------------------------
# 单元测试 (直接 python dominant_cluster.py 运行)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 构造测试数据：在 (200, 0) 附近聚了 8 个点，外加 2 个散点
    np.random.seed(42)
    cluster_pts = np.random.randn(8, 6) * 2.0 + [200, 0, 50, 180, 0, 45]
    noise_pts = np.array([
        [250, 80, 55, 180, 0, 30],
        [180, -50, 48, 180, 0, 60],
    ])
    cands = np.vstack([cluster_pts, noise_pts]).tolist()

    goal, cnt = dominant_cluster(cands, bin_mm=5)
    print(f"Candidates: {len(cands)}")
    print(f"Cluster frames: {cnt}/{len(cands)}")
    print(f"Centroid: X={goal[0]:.1f} Y={goal[1]:.1f} Z={goal[2]:.1f} Yaw={goal[5]:.1f} deg")
    print("Expected: X~200 Y~0, cluster~8 frames")
