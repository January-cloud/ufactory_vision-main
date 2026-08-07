#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端视觉抓取流程模拟
======================
从本地深度图出发，完整走一遍视觉抓取推理管线：

  深度图 (HxW float32, 米)
    -> GGCNN2 神经网络推理
    -> 抓取点提取 (相机坐标系 [x,y,z,ang,width])
    -> 坐标变换 (相机系 -> 基坐标系)
    -> 候选聚类 (多帧稳定)
    -> TaskBuilder 生成 pick-and-place 序列
    -> 输出 JSON (与 POST /task 格式一致)

用法:
    # 使用程序生成的合成深度图 (内置, 免硬件)
    python e2e_pipeline.py

    # 多帧模拟 (模拟相机连续拍摄, 每帧微调噪声)
    python e2e_pipeline.py --frames 20

    # 从文件加载真实深度图
    python e2e_pipeline.py --depth depth.npy

    # 从文件加载 16-bit PNG 深度图 (mm 单位)
    python e2e_pipeline.py --depth-png depth.png

    # 导出最终任务 JSON
    python e2e_pipeline.py --export task.json
"""

import os
import sys
import json
import time
import math
import argparse
import logging
import numpy as np
import cv2

# 把上级目录加入搜索路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.join(_current_dir, '..')
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from grasp.ggcnn_torch import TorchGGCNN
from grasp.helpers.matrix_funcs import euler2mat, convert_pose
from multi_arm.dominant_cluster import dominant_cluster
from simulation.task_builder import TaskBuilder, SimGraspConfig

# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数 (与 run_simulation.py 一致)
# ═══════════════════════════════════════════════════════════════════════════════

CAM_WIDTH, CAM_HEIGHT = 640, 480

# 相机内参 (默认 D435 典型值)
CAM_FX, CAM_FY = 615.0, 615.0

# 手眼标定 (相机在法兰系的位姿, xyz米 + rpy弧度)
EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456,
                           -0.004202176, -0.00848499, 1.5898775]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# 关键位姿 (mm)
DETECT_XYZ = [200.0, 0.0, 380.0]         # 观察位
RELEASE_XYZ = [225.0, -89.0, 83.0]       # 释放点
ABOVE_Z = 300.0                           # 安全悬停高度
GRASPING_MIN_Z = 70                       # 最低 Z 限位
GRIPPER_Z_MM = 55                         # 吸盘长度补偿
GRASPING_RANGE = [180, 350, -200, 200]   # 安全抓取区

# 聚类参数
CAND_WINDOW = 12
CAND_BIN_MM = 8
CAND_MIN_FRAMES = 6

# 模型路径
MODEL_FILE = os.path.join(_parent_dir, 'models', 'epoch_50_cornell')


# ═══════════════════════════════════════════════════════════════════════════════
# 合成深度图生成器
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_depth(width=640, height=480,
                         table_depth_m=0.5,      # 桌面深度 (米)
                         obj_x=320, obj_y=240,    # 物体在图像中的位置 (像素)
                         obj_size=60,              # 物体大小 (像素)
                         obj_height_m=0.03,       # 物体高度 (米, 桌面之上)
                         noise_std_m=0.001,        # 深度噪声标准差 (米)
                         seed=None):
    """生成一张模拟桌面+物块的合成深度图。

    场景: 相机朝下拍摄桌面，桌面上有一个矩形物体。
    物体处的深度 = table_depth_m - obj_height_m (比桌面更靠近相机)。
    """
    rng = np.random.RandomState(seed)

    # 桌面背景
    depth = np.full((height, width), table_depth_m, dtype=np.float32)

    # 在图像中心附近放置一个矩形物体
    half = obj_size // 2
    y1 = max(0, obj_y - half)
    y2 = min(height, obj_y + half)
    x1 = max(0, obj_x - half)
    x2 = min(width, obj_x + half)
    depth[y1:y2, x1:x2] = table_depth_m - obj_height_m

    # 高斯噪声
    noise = rng.randn(height, width).astype(np.float32) * noise_std_m
    depth += noise

    # 物体边缘轻微模糊 (模拟真实相机)
    depth = cv2.GaussianBlur(depth, (5, 5), 1.0)

    return depth


def make_synthetic_depth_multi_obj(width=640, height=480,
                                    table_depth_m=0.5,
                                    objects=None,
                                    noise_std_m=0.001,
                                    seed=None):
    """生成多物体的合成深度图。

    参数:
        objects: [(cx, cy, size, height_m), ...] 物体列表
    """
    rng = np.random.RandomState(seed)
    depth = np.full((height, width), table_depth_m, dtype=np.float32)

    if objects is None:
        objects = [
            (280, 220, 50, 0.025),   # (cx, cy, size, height)
            (380, 260, 45, 0.030),
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


# ═══════════════════════════════════════════════════════════════════════════════
# 深度图加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_depth_npy(path):
    """加载 .npy 格式深度图 (float32, 米)。"""
    arr = np.load(path)
    return arr.astype(np.float32)


def load_depth_png(path, depth_scale=0.001):
    """加载 16-bit PNG 深度图 (mm 单位), 转为 float32 米。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取深度图: {path}")
    depth = img.astype(np.float32) * depth_scale
    depth[depth <= 0] = np.nan
    return depth


# ═══════════════════════════════════════════════════════════════════════════════
# 坐标变换 (与 run_simulation.py 一致)
# ═══════════════════════════════════════════════════════════════════════════════

def get_sim_eef_pose_m():
    """模拟末端位姿 (观察位, 相机朝下)。"""
    return [
        DETECT_XYZ[0] * 0.001,
        DETECT_XYZ[1] * 0.001,
        DETECT_XYZ[2] * 0.001,
        math.pi, 0.0, 0.0,
    ]


def cam_result_to_base(result):
    """相机系抓取点 -> 基坐标系目标 (mm, 度)。"""
    x, y, z, ang = result[0], result[1], result[2], result[3]
    gp = [x, y, z, 0, 0, -1 * ang]

    eef_pose = get_sim_eef_pose_m()
    mat = (euler2mat(eef_pose)
           * euler2mat(EULER_EEF_TO_COLOR_OPT)
           * euler2mat(EULER_COLOR_TO_DEPTH_OPT))
    gp_base = convert_pose(gp, mat)

    # 角度归一化
    if gp_base[5] < -np.pi:
        gp_base[5] += np.pi
    elif gp_base[5] > 0:
        gp_base[5] -= np.pi

    return [
        gp_base[0] * 1000,
        gp_base[1] * 1000,
        gp_base[2] * 1000 + GRIPPER_Z_MM,
        180, 0,
        math.degrees(gp_base[5] + np.pi / 2),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(depth_image, K, ggcnn, builder,
                 n_frames=12, noise_std=0.0, seed=42):
    """执行完整的视觉抓取推理管线。

    参数:
        depth_image: (H, W) float32 深度图 (米)
        K:           (3,3) 相机内参矩阵
        ggcnn:       TorchGGCNN 实例
        builder:     TaskBuilder 实例
        n_frames:    模拟帧数 (用于聚类稳定)
        noise_std:   每帧添加的噪声标准差 (米), 0=不加噪
        seed:        随机种子

    返回:
        dict, 包含完整的管线输出
    """
    rng = np.random.RandomState(seed)
    H, W = depth_image.shape
    eef_pose = get_sim_eef_pose_m()

    # ── 阶段 1: 逐帧 GGCNN 推理 ──
    candidates = []
    all_results = []
    grasp_img_last = None

    for frame_i in range(n_frames):
        # 添加帧间噪声 (模拟真实相机抖动)
        if noise_std > 0:
            frame_depth = depth_image + rng.randn(H, W).astype(np.float32) * noise_std
        else:
            frame_depth = depth_image.copy()

        grasp_img, result = ggcnn.get_grasp_img(frame_depth, K, eef_pose[2])
        grasp_img_last = grasp_img

        if result is not None:
            z_mm = result[2] * 1000  # 深度转为 mm
            if z_mm > 200:  # 有效识别最小距离
                cand = cam_result_to_base(result)
                candidates.append(cand)
                all_results.append({
                    "frame": frame_i + 1,
                    "cam_xyz_m": [round(result[0], 4), round(result[1], 4), round(result[2], 4)],
                    "cam_ang_deg": round(math.degrees(result[3]), 1),
                    "cam_width_mm": round(result[4], 1),
                    "depth_center_mm": round(result[5], 1),
                    "base_xyz_mm": [round(cand[0], 1), round(cand[1], 1), round(cand[2], 1)],
                    "base_yaw_deg": round(cand[5], 1),
                })

    # ── 阶段 2: 候选聚类 ──
    goal = None
    cluster_cnt = 0
    stable = False

    if len(candidates) >= 2:
        goal, cluster_cnt = dominant_cluster(candidates, CAND_BIN_MM)
        stable = cluster_cnt >= CAND_MIN_FRAMES
    elif len(candidates) == 1:
        goal = candidates[0]
        cluster_cnt = 1

    # ── 阶段 3: 构建任务序列 ──
    task_sequence = None
    in_range = False
    if goal is not None:
        in_range = (GRASPING_RANGE[0] <= goal[0] <= GRASPING_RANGE[1] and
                    GRASPING_RANGE[2] <= goal[1] <= GRASPING_RANGE[3])
        if in_range:
            task_sequence = builder.build_pick_and_place(list(goal))

    return {
        "pipeline_config": {
            "detect_xyz_mm": DETECT_XYZ,
            "release_xyz_mm": RELEASE_XYZ,
            "above_z_mm": ABOVE_Z,
            "grasping_min_z_mm": GRASPING_MIN_Z,
            "grasping_range_mm": GRASPING_RANGE,
            "model_file": MODEL_FILE,
            "cam_intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
        },
        "input": {
            "depth_shape": list(depth_image.shape),
            "depth_min_m": round(float(np.nanmin(depth_image)), 4),
            "depth_max_m": round(float(np.nanmax(depth_image)), 4),
            "n_frames": n_frames,
            "noise_std_m": noise_std,
        },
        "inference": {
            "frames_processed": n_frames,
            "frames_with_detection": len(all_results),
            "detection_rate": f"{len(all_results)}/{n_frames}",
            "per_frame_results": all_results,
        },
        "clustering": {
            "candidates_collected": len(candidates),
            "cluster_bin_mm": CAND_BIN_MM,
            "cluster_frames": cluster_cnt,
            "min_frames_for_lock": CAND_MIN_FRAMES,
            "stable_locked": stable,
        },
        "grasp_target": {
            "goal": [round(v, 1) for v in goal] if goal else None,
            "in_safe_range": in_range,
        },
        "task_sequence": task_sequence,
        "task_sequence_json": json.dumps({"sequence": task_sequence}, indent=2, ensure_ascii=False)
        if task_sequence else None,
        "grasp_heatmap": grasp_img_last,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════════════

def print_pipeline_result(result):
    """格式化打印管线输出。"""
    cfg = result["pipeline_config"]
    inp = result["input"]
    inf = result["inference"]
    clu = result["clustering"]
    gt = result["grasp_target"]

    width = 64

    print()
    print("=" * width)
    print("  端到端视觉抓取 — 管线输出")
    print("=" * width)

    # ── 输入 ──
    print(f"\n  [输入]")
    print(f"    深度图尺寸:  {inp['depth_shape'][1]} x {inp['depth_shape'][0]} (WxH)")
    print(f"    深度范围:    {inp['depth_min_m']:.3f} ~ {inp['depth_max_m']:.3f} m")
    print(f"    模拟帧数:    {inp['n_frames']}")
    print(f"    帧间噪声:    {inp['noise_std_m']*1000:.2f} mm std")

    # ── 相机配置 ──
    print(f"\n  [配置]")
    print(f"    相机内参:    fx={cfg['cam_intrinsics']['fx']:.1f} fy={cfg['cam_intrinsics']['fy']:.1f} "
          f"cx={cfg['cam_intrinsics']['cx']:.1f} cy={cfg['cam_intrinsics']['cy']:.1f}")
    print(f"    观察位:      X={cfg['detect_xyz_mm'][0]:.0f} Y={cfg['detect_xyz_mm'][1]:.0f} "
          f"Z={cfg['detect_xyz_mm'][2]:.0f} mm")
    print(f"    释放点:      X={cfg['release_xyz_mm'][0]:.0f} Y={cfg['release_xyz_mm'][1]:.0f} "
          f"Z={cfg['release_xyz_mm'][2]:.0f} mm")
    print(f"    安全悬停高度: {cfg['above_z_mm']:.0f} mm")
    print(f"    安全抓取区:   X=[{cfg['grasping_range_mm'][0]:.0f}, {cfg['grasping_range_mm'][1]:.0f}] "
          f"Y=[{cfg['grasping_range_mm'][2]:.0f}, {cfg['grasping_range_mm'][3]:.0f}] mm")

    # ── 推理 ──
    print(f"\n  [GGCNN 推理]")
    print(f"    总帧数:        {inf['frames_processed']}")
    print(f"    有效检测帧:    {inf['frames_with_detection']} ({inf['detection_rate']})")
    if inf["per_frame_results"]:
        first = inf["per_frame_results"][0]
        last = inf["per_frame_results"][-1]
        print(f"    第 1 帧:       相机系 ({first['cam_xyz_m'][0]:.4f}, {first['cam_xyz_m'][1]:.4f}, "
              f"{first['cam_xyz_m'][2]:.4f}) m  ang={first['cam_ang_deg']:.1f} deg")
        print(f"                   基系 ({first['base_xyz_mm'][0]:.1f}, {first['base_xyz_mm'][1]:.1f}, "
              f"{first['base_xyz_mm'][2]:.1f}) mm  yaw={first['base_yaw_deg']:.1f} deg")
        if len(inf["per_frame_results"]) > 1:
            print(f"    第 {len(inf['per_frame_results'])} 帧:      "
                  f"相机系 ({last['cam_xyz_m'][0]:.4f}, {last['cam_xyz_m'][1]:.4f}, "
                  f"{last['cam_xyz_m'][2]:.4f}) m  ang={last['cam_ang_deg']:.1f} deg")

    # ── 聚类 ──
    print(f"\n  [候选聚类]")
    print(f"    候选数:        {clu['candidates_collected']}")
    print(f"    聚类网格:      {clu['cluster_bin_mm']} mm")
    print(f"    主簇帧数:      {clu['cluster_frames']} / {clu['min_frames_for_lock']} (锁定阈值)")
    print(f"    稳定锁定:      {'YES' if clu['stable_locked'] else 'NO'}")

    # ── 抓取目标 ──
    print(f"\n  [抓取目标]")
    if gt["goal"]:
        g = gt["goal"]
        print(f"    基坐标系:      X={g[0]:.1f} Y={g[1]:.1f} Z={g[2]:.1f} mm")
        print(f"    姿态:          Roll={g[3]:.0f} Pitch={g[4]:.0f} Yaw={g[5]:.1f} deg")
        print(f"    安全区内:      {'YES' if gt['in_safe_range'] else 'NO (跳过抓取)'}")

        # 检查 Z 限位
        grasp_z = max(g[2], GRASPING_MIN_Z)
        if g[2] < GRASPING_MIN_Z:
            print(f"    Z 限位修正:    {g[2]:.1f} -> {grasp_z:.1f} mm (安全限位 {GRASPING_MIN_Z} mm)")
    else:
        print(f"    无有效目标")

    # ── 任务序列 ──
    print(f"\n  [任务序列]")
    if result["task_sequence"]:
        seq = result["task_sequence"]
        print(f"    总步数:        {len(seq)}")
        total_wait = sum(s.get("wait", 0) for s in seq)
        print(f"    总等待时间:    {total_wait:.1f}s")
        print(f"\n    {'─'*52}")
        print(f"    {'步':<4} {'类型':<10} {'目标位置 / 动作':<38}")
        print(f"    {'─'*52}")
        for i, step in enumerate(seq):
            t = step["type"]
            if t == "move":
                p = step["params"]
                print(f"    {i+1:<4} {t:<10} "
                      f"({p['x']:6.0f}, {p['y']:6.0f}, {p['z']:6.0f}) mm  "
                      f"yaw={p.get('yaw', 0):5.0f} deg  wait={step['wait']}s")
            elif t == "vacuum":
                on_off = "ON " if step["params"]["on"] else "OFF"
                print(f"    {i+1:<4} {t:<10} VACUUM {on_off}"
                      f"{'':>26}  wait={step['wait']}s")
        print(f"    {'─'*52}")
    else:
        if gt["goal"] and not gt["in_safe_range"]:
            print(f"    目标超出安全区, 未生成序列")
        else:
            print(f"    无目标, 未生成序列")

    print()
    print("=" * width)

    # 输出完整 JSON
    if result["task_sequence_json"]:
        print(f"\n  [POST /task JSON]")
        print(result["task_sequence_json"])


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='端到端视觉抓取流程模拟',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python e2e_pipeline.py                        # 合成深度图, 单帧
  python e2e_pipeline.py --frames 20            # 20 帧模拟 (含聚类)
  python e2e_pipeline.py --depth depth.npy      # 从 .npy 文件加载深度图
  python e2e_pipeline.py --depth-png d.png      # 从 PNG 文件加载深度图
  python e2e_pipeline.py --export task.json     # 导出任务 JSON
  python e2e_pipeline.py --show                 # 显示 GGCNN 热力图窗口
  python e2e_pipeline.py --multi-obj            # 多物体场景
        """
    )
    parser.add_argument('--depth', type=str, default=None,
                        help='深度图 .npy 文件 (float32, 米)')
    parser.add_argument('--depth-png', type=str, default=None,
                        help='深度图 PNG 文件 (16-bit, mm)')
    parser.add_argument('--frames', '-n', type=int, default=12,
                        help='模拟帧数 (默认 12)')
    parser.add_argument('--noise', type=float, default=0.0003,
                        help='帧间噪声标准差, 米 (默认 0.0003)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--multi-obj', action='store_true',
                        help='生成多物体合成深度图')
    parser.add_argument('--model', type=str, default=MODEL_FILE,
                        help=f'GGCNN2 模型权重路径 (默认: {MODEL_FILE})')
    parser.add_argument('--export', '-e', type=str, default=None,
                        help='导出最终任务 JSON 到文件')
    parser.add_argument('--show', '-s', action='store_true',
                        help='显示 GGCNN 热力图窗口')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='仅输出最终 JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    # ═════════════════════════════════════════════════════════════════════════
    # 1. 准备深度图
    # ═════════════════════════════════════════════════════════════════════════
    if args.depth:
        depth_image = load_depth_npy(args.depth)
    elif args.depth_png:
        depth_image = load_depth_png(args.depth_png)
        # PNG 深度图通常按图像宽高存储, 取实际尺寸作为相机分辨率
        global CAM_WIDTH, CAM_HEIGHT
        CAM_HEIGHT, CAM_WIDTH = depth_image.shape[:2]
    else:
        # 生成合成深度图
        if args.multi_obj:
            depth_image = make_synthetic_depth_multi_obj(
                CAM_WIDTH, CAM_HEIGHT, table_depth_m=0.5, seed=args.seed
            )
        else:
            depth_image = make_synthetic_depth(
                CAM_WIDTH, CAM_HEIGHT,
                table_depth_m=0.5,
                obj_x=320, obj_y=240,
                obj_size=60,
                obj_height_m=0.03,
                seed=args.seed,
            )

    # 更新相机的全局分辨率 (确保与深度图匹配)
    CAM_HEIGHT, CAM_WIDTH = depth_image.shape[:2]

    # 相机内参
    fx = CAM_FX * (CAM_WIDTH / 640.0)
    fy = CAM_FY * (CAM_HEIGHT / 480.0)
    cx, cy = CAM_WIDTH / 2.0, CAM_HEIGHT / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # ═════════════════════════════════════════════════════════════════════════
    # 2. 初始化 GGCNN
    # ═════════════════════════════════════════════════════════════════════════
    from queue import Queue
    ggcnn = TorchGGCNN({
        'MODEL_FILE': args.model,
        'OPEN_LOOP_HEIGHT': 0,
        'GGCNN_IN_THREAD': False,
        'DEPTH_CAM_K': K,
    }, Queue(1), Queue(1))
    time.sleep(0.5)  # 等待模型加载

    # ═════════════════════════════════════════════════════════════════════════
    # 3. 初始化 TaskBuilder
    # ═════════════════════════════════════════════════════════════════════════
    config = SimGraspConfig(
        detect_xyz=list(DETECT_XYZ),
        release_xyz=list(RELEASE_XYZ),
        above_z=ABOVE_Z,
        grasping_min_z=GRASPING_MIN_Z,
    )
    builder = TaskBuilder(config)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. 执行管线
    # ═════════════════════════════════════════════════════════════════════════
    result = run_pipeline(
        depth_image, K, ggcnn, builder,
        n_frames=args.frames,
        noise_std=args.noise,
        seed=args.seed,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 5. 输出
    # ═════════════════════════════════════════════════════════════════════════
    if args.quiet:
        if result["task_sequence_json"]:
            print(result["task_sequence_json"])
        else:
            print(json.dumps({"sequence": [], "error": "no valid grasp target"}))
    else:
        print_pipeline_result(result)

    # 导出
    if args.export and result["task_sequence"]:
        export_data = {"sequence": result["task_sequence"]}
        with open(args.export, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n[EXPORT] 任务序列已导出到: {args.export}")

    # 显示热力图
    if args.show and result["grasp_heatmap"] is not None:
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        combined = np.hstack([depth_display, result["grasp_heatmap"]])
        cv2.imshow("E2E Pipeline - Depth (left) / GGCNN Heatmap (right)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
