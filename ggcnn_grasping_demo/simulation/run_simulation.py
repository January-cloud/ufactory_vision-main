#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真模式 · 单臂视觉抓取
========================
与 run_rs_d435_grasp_lite6_new_best.py 对应，将物理机械臂控制替换为
HTTP 仿真调用。相机固定于观察位，识别目标后构建完整 pick-and-place
序列并 POST 到仿真服务器执行。

流程:
  启动时交互选择识别方式（摄像头 / 手动输入）
  → 摄像头识别: 观察位拍照 → GGCNN 推理 → 候选缓冲+聚类 → LOCKED
  → 手动输入: HTTP 接收外部坐标 → 校验 → LOCKED
  → TaskBuilder 构建序列 → SimulationClient POST → 等待冷却 → 循环
操作：放物块即自动识别并发送仿真任务，按 q / ESC 退出。

识别方式（工序开始时可交互选择）:
  [1] 摄像头识别: GGCNN 神经网络从深度图推理
  [2] 手动坐标输入: HTTP 服务器接收外部坐标 (POST /grasp_target)
  --ext-input: 跳过交互选择，直接进入手动输入模式

摄像头模式:
  默认: BuiltinCamera (本地合成图像，无需任何硬件或服务器)
  --sim-camera: SimCamera (从 HTTP 仿真服务器获取图像)
  --no-camera: 静态虚拟图像 (测试用)
"""

import os
import sys
import cv2
import math
import time
import json
import logging
import argparse
import numpy as np
from queue import Queue
from collections import Counter
import enum

# 把上级目录（ggcnn_grasping_demo）加入搜索路径，才能 import 到 camera / grasp / multi_arm 模块
_sim_dir = os.path.dirname(os.path.abspath(__file__))
_demo_dir = os.path.join(_sim_dir, '..')
if _demo_dir not in sys.path:
    sys.path.insert(0, _demo_dir)
from camera.utils import get_combined_img
from grasp.ggcnn_torch import TorchGGCNN
from grasp.helpers.matrix_funcs import euler2mat, convert_pose

# simulation 包内的模块
from simulation.simulation_client import SimulationClient, SimulationClientError
from simulation.task_builder import TaskBuilder, SimGraspConfig
from simulation.sim_camera import SimCamera
from simulation.builtin_camera import BuiltinCamera
from simulation.external_input import ExternalInputServer

# 引入 dominant_cluster（位于 multi_arm 包内）
from multi_arm.dominant_cluster import dominant_cluster


class RecognitionMode(enum.Enum):
    """抓取目标识别模式：工序开始时由操作人员交互选择。"""
    CAMERA = "camera"   # 摄像头识别 (GGCNN 神经网络推理)
    MANUAL = "manual"   # 手动坐标输入 (HTTP 服务器接收外部坐标)


# ═══════════════════════════════════════════════════════════════════════════════
# 配置区（与 run_rs_d435_grasp_lite6_new_best.py 保持一致）
# ═══════════════════════════════════════════════════════════════════════════════

# ── 仿真服务器 ──
SIM_SERVER_URL = "http://192.168.1.121:8080"
SIM_TIMEOUT = 15.0         # HTTP 请求超时 (s)
SIM_RETRIES = 2            # 失败重试次数

# ── 相机 ──
CAM_WIDTH, CAM_HEIGHT = 640, 480
WIN_NAME = 'RealSense-D435 SIMULATION'

# ── GGCNN 模型 ──
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'epoch_50_cornell')

# ── 手眼标定（与真实机械臂脚本一致） ──
EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456,
                           -0.004202176, -0.00848499, 1.5898775]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

# ── 关键位姿 ──
DETECT_XYZ = [200.0, 0.0, 380.0]         # 观察位（相机在此拍照）
RELEASE_XYZ = [225.0, -89.0, 83.0]       # 放置/释放点
GRASPING_RANGE = [180, 350, -200, 200]   # 抓取安全区 [x_min, x_max, y_min, y_max]

# ── 抓取参数 ──
GRIPPER_Z_MM = 55                         # 吸盘长度补偿 (法兰→吸盘底面)
GRASPING_MIN_Z = 70                       # 最低 Z 限位 (防撞桌)
ABOVE_Z = 300.0                           # 安全悬停高度
MIN_RESULT_Z_MM = 200                     # 相机有效识别最小距离

# ── 聚类与稳定性 ──
CAND_WINDOW = 12                          # 候选缓冲窗口帧数
CAND_BIN_MM = 8                           # 物块聚类网格 (mm)
CAND_MIN_FRAMES = 6                       # 主簇最少帧数（触发 LOCKED）

# ── 冷却 ──
COOLDOWN_SEC = 1.0                        # 两轮抓取最小间隔

# ── 仿真动作等待时间 ──
MOVE_WAIT = 0.5                           # move 后等待 (s)
DESCEND_WAIT = 1.0                        # 下降后等待 (s)
SUCTION_WAIT = 0.8                        # 吸取后等待 (s)
RELEASE_WAIT = 0.5                        # 释放后等待 (s)


# ═══════════════════════════════════════════════════════════════════════════════
# 坐标变换（与原始脚本相同的算法，使用模拟的末端位姿）
# ═══════════════════════════════════════════════════════════════════════════════

def get_sim_eef_pose_m():
    """返回模拟的末端位姿（观察位）。

    仿真模式下无真实机械臂连接，使用观察位作为"虚拟末端位姿"，
    姿态为相机朝下 (roll=pi)。

    返回:
        [x, y, z, roll, pitch, yaw] — 米, 弧度
    """
    return [
        DETECT_XYZ[0] * 0.001,
        DETECT_XYZ[1] * 0.001,
        DETECT_XYZ[2] * 0.001,
        math.pi,    # roll=180° 相机朝下
        0.0,        # pitch
        0.0,        # yaw
    ]


def cam_result_to_base(result):
    """将 GGCNN 相机系抓取点转换到机械臂基坐标系。

    变换链: Base ← EEF(模拟观察位) ← ColorCamera(手眼标定) ← DepthCamera

    参数:
        result: [x, y, z, ang, width, depth_center] — 相机坐标系

    返回:
        [X, Y, Z, 180, 0, Yaw] — mm, 度（基坐标系）
    """
    x, y, z, ang = result[0], result[1], result[2], result[3]
    gp = [x, y, z, 0, 0, -1 * ang]

    eef_pose = get_sim_eef_pose_m()
    mat = (euler2mat(eef_pose)
           * euler2mat(EULER_EEF_TO_COLOR_OPT)
           * euler2mat(EULER_COLOR_TO_DEPTH_OPT))
    gp_base = convert_pose(gp, mat)

    # 抓取角度归一化到 [-pi, 0]
    if gp_base[5] < -np.pi:
        gp_base[5] += np.pi
    elif gp_base[5] > 0:
        gp_base[5] -= np.pi

    return [
        gp_base[0] * 1000,
        gp_base[1] * 1000,
        gp_base[2] * 1000 + GRIPPER_Z_MM,
        180,
        0,
        math.degrees(gp_base[5] + np.pi / 2),
    ]


def validate_grasp_target(target, logger=None):
    """全面校验抓取目标坐标是否合法、可达。

    对传入的基坐标系目标 [X, Y, Z, Roll, Pitch, Yaw] (mm/度) 执行:
      - XY 工作区范围检查
      - Z 高度上下限检查
      - 姿态角度范围检查与归一化
      - 非数值 / 无穷大检测

    参数:
        target: [x, y, z, roll, pitch, yaw] 或包含至少 6 个元素的 list
        logger: 可选的 logging.Logger，用于输出警告信息

    返回:
        (is_valid, corrected_target, reason)
        is_valid:        bool — True 表示目标可用
        corrected_target: list — 校验通过的目标（yaw 可能已被归一化）
        reason:          str  — 校验失败时的原因描述，成功时为 "OK"
    """
    # ── 基本类型检查 ──
    if target is None or len(target) < 3:
        return False, target, "目标为空或缺少必要字段 (至少需要 X, Y, Z)"

    try:
        x, y, z = float(target[0]), float(target[1]), float(target[2])
        roll = float(target[3]) if len(target) > 3 else 180.0
        pitch = float(target[4]) if len(target) > 4 else 0.0
        yaw = float(target[5]) if len(target) > 5 else 0.0
    except (TypeError, ValueError) as e:
        return False, target, f"坐标值无法转换为数字: {e}"

    # ── 非数值 / 无穷大检测 ──
    for name, val in [('X', x), ('Y', y), ('Z', z), ('Roll', roll),
                       ('Pitch', pitch), ('Yaw', yaw)]:
        if math.isnan(val):
            return False, target, f"{name} 为 NaN (非数值)"
        if math.isinf(val):
            return False, target, f"{name} 为无穷大"

    # ── XY 工作区范围 ──
    x_min, x_max = GRASPING_RANGE[0], GRASPING_RANGE[1]
    y_min, y_max = GRASPING_RANGE[2], GRASPING_RANGE[3]
    if not (x_min <= x <= x_max):
        return False, target, f"X={x:.1f} 超出工作区 [{x_min}, {x_max}] mm"
    if not (y_min <= y <= y_max):
        return False, target, f"Y={y:.1f} 超出工作区 [{y_min}, {y_max}] mm"

    # ── Z 高度检查 ──
    z_max = ABOVE_Z + 100  # 安全悬停高度以上 100mm 为上限
    if z < GRASPING_MIN_Z:
        return False, target, f"Z={z:.1f} 低于最低限位 {GRASPING_MIN_Z} mm (防撞桌)"
    if z > z_max:
        return False, target, f"Z={z:.1f} 高于最大允许高度 {z_max} mm"

    # ── 姿态角度校验与归一化 ──
    # Roll 应为 ~180° (末端朝下)，允许 ±10° 偏差
    if not (170.0 <= roll <= 190.0):
        if logger:
            logger.warning("Roll=%.1f° 偏离预期 (180° 末端朝下)，已自动修正为 180°", roll)
        roll = 180.0

    # Pitch 应为 ~0°，允许 ±15° 偏差
    if not (-15.0 <= pitch <= 15.0):
        if logger:
            logger.warning("Pitch=%.1f° 偏离预期 (0°)，已自动修正为 0°", pitch)
        pitch = 0.0

    # Yaw 归一化到 [-180, 180]
    yaw = yaw % 360.0
    if yaw > 180.0:
        yaw -= 360.0
    elif yaw < -180.0:
        yaw += 360.0

    if abs(yaw) > 90.0:
        if logger:
            logger.warning("Yaw=%.1f° 超出抓取角范围 [-90°, 90°]，已钳位", yaw)
        yaw = max(-90.0, min(90.0, yaw))

    corrected = [x, y, z, roll, pitch, yaw]
    return True, corrected, "OK"


def in_range(g):
    """目标 XY 是否在安全抓取范围内。"""
    return (GRASPING_RANGE[0] <= g[0] <= GRASPING_RANGE[1] and
            GRASPING_RANGE[2] <= g[1] <= GRASPING_RANGE[3])


def choose_recognition_mode(args, logger):
    """让操作人员在工序开始时选择物体识别方式。

    交互菜单:
        [1] 摄像头识别 (GGCNN 神经网络从深度图推理抓取点)
        [2] 手动坐标输入 (HTTP 服务器接收外部坐标 POST /grasp_target)

    CLI 兼容:
        --ext-input: 跳过交互提示，直接进入手动坐标输入模式（非交互场景）

    返回:
        RecognitionMode 枚举值
    """
    # ── 非交互覆盖：--ext-input 直接进入手动模式 ──
    if args.ext_input:
        logger.info("--ext-input 已指定，跳过交互选择，使用手动坐标输入模式")
        return RecognitionMode.MANUAL

    print("\n" + "=" * 60)
    print("  物体识别方式选择 / Recognition Mode Selection")
    print("=" * 60)
    print("  [1] 摄像头识别 (Camera recognition)")
    print("      GGCNN 神经网络从深度图推理抓取点")
    print()
    print("  [2] 手动坐标输入 (Manual coordinate input)")
    print("      HTTP 服务器接收外部坐标 (POST /grasp_target)")
    print("=" * 60)

    while True:
        try:
            choice = input("请选择 (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("  输入中断，默认使用摄像头识别模式")
            logger.warning("交互输入中断，默认使用摄像头识别模式")
            return RecognitionMode.CAMERA

        if choice == '1':
            logger.info("已选择: 摄像头识别 (GGCNN)")
            return RecognitionMode.CAMERA
        elif choice == '2':
            logger.info("已选择: 手动坐标输入 (HTTP)")
            return RecognitionMode.MANUAL
        else:
            print("  无效输入，请输入 1 或 2")

    return RecognitionMode.CAMERA


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 解析命令行参数 ──
    parser = argparse.ArgumentParser(
        description='仿真模式 · 视觉引导抓取 (单臂)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_simulation.py                              # 默认：本地合成摄像头
  python run_simulation.py --server http://192.168.1.121:8080
  python run_simulation.py --sim-camera                 # 从仿真服务器获取图像
  python run_simulation.py --no-camera                  # 无相机测试模式
  python run_simulation.py --ext-input                  # 启用外部坐标输入
  python run_simulation.py --ext-input --ext-port 9090  # 自定义外部输入端口
        """
    )
    parser.add_argument('--server', default=SIM_SERVER_URL,
                        help=f'仿真服务器 URL (默认: {SIM_SERVER_URL})')
    parser.add_argument('--model', default=MODEL_FILE,
                        help=f'GGCNN2 模型权重路径 (默认: 自动查找)')
    parser.add_argument('--timeout', type=float, default=SIM_TIMEOUT,
                        help=f'HTTP 超时秒数 (默认: {SIM_TIMEOUT})')
    parser.add_argument('--no-camera', action='store_true',
                        help='无相机模式（使用虚拟深度图，用于测试通信链路）')
    parser.add_argument('--sim-camera', action='store_true',
                        help='从仿真服务器获取摄像头图像（POST /get_camera）')
    parser.add_argument('--cam-fx', type=float, default=None,
                        help='摄像头焦距 fx (像素)，默认 615')
    parser.add_argument('--cam-fy', type=float, default=None,
                        help='摄像头焦距 fy (像素)，默认 615')
    parser.add_argument('--ext-input', action='store_true',
                        help='启用外部坐标输入 HTTP 服务器 (POST /grasp_target)')
    parser.add_argument('--ext-port', type=int, default=8090,
                        help='外部坐标输入 HTTP 服务器端口 (默认: 8090)')
    parser.add_argument('--ext-host', type=str, default='0.0.0.0',
                        help='外部坐标输入 HTTP 服务器绑定地址 (默认: 0.0.0.0)')
    args = parser.parse_args()

    # ── 日志配置 ──
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('simulation')

    logger.info("=" * 60)
    logger.info("仿真模式 · 视觉引导抓取 启动中...")
    logger.info(f"  仿真服务器: {args.server}")
    logger.info(f"  模型路径:   {args.model}")
    logger.info(f"  观察位:     {DETECT_XYZ}")
    logger.info(f"  释放位:     {RELEASE_XYZ}")
    logger.info("=" * 60)

    # ═════════════════════════════════════════════════════════════════════════
    # 0. 选择物体识别方式（交互菜单 / --ext-input 覆盖）
    # ═════════════════════════════════════════════════════════════════════════
    mode = choose_recognition_mode(args, logger)
    mode_desc = ("摄像头识别 (GGCNN)" if mode == RecognitionMode.CAMERA
                 else "手动坐标输入 (HTTP)")
    logger.info("识别模式: %s", mode_desc)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. 初始化仿真客户端
    # ═════════════════════════════════════════════════════════════════════════
    client = SimulationClient(
        base_url=args.server,
        timeout=args.timeout,
        retries=SIM_RETRIES,
    )

    logger.info("检查仿真服务器连通性...")
    if not client.check_connection():
        logger.error(
            "无法连接到仿真服务器 %s，请确认服务器已启动。"
            "使用 --server 指定正确的地址。",
            args.server
        )
        client.close()
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. 初始化相机
    # ═════════════════════════════════════════════════════════════════════════
    if args.no_camera:
        logger.warning("无相机模式：将使用虚拟深度图（仅用于测试通信链路）")
        camera = None
        K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]])
    elif args.sim_camera:
        logger.info("初始化仿真摄像头 (HTTP /get_camera)...")
        cam_fx = args.cam_fx if args.cam_fx else 615.0
        cam_fy = args.cam_fy if args.cam_fy else 615.0
        camera = SimCamera(
            client, width=CAM_WIDTH, height=CAM_HEIGHT,
            fx=cam_fx, fy=cam_fy,
        )
        ci, _ = camera.get_intrinsics()
        K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
        logger.info(f"仿真相机内参: fx={ci.fx:.1f} fy={ci.fy:.1f} cx={ci.ppx:.1f} cy={ci.ppy:.1f}")
        # 测试获取一帧，确认摄像头端点可用
        try:
            test_color, test_depth = camera.get_images()
            logger.info(
                "仿真摄像头测试成功: color=%s depth=%s",
                test_color.shape, test_depth.shape
            )
        except Exception as e:
            logger.error("仿真摄像头测试失败: %s", e)
            logger.error("请确认仿真服务器 /get_camera 端点正常，或改用 --no-camera")
            client.close()
            sys.exit(1)
    else:
        # 默认：本地合成摄像头（无需任何硬件或外部服务器）
        logger.info("初始化本地合成摄像头 (BuiltinCamera)...")
        cam_fx = args.cam_fx if args.cam_fx else 615.0
        cam_fy = args.cam_fy if args.cam_fy else 615.0
        camera = BuiltinCamera(
            width=CAM_WIDTH, height=CAM_HEIGHT,
            fx=cam_fx, fy=cam_fy,
        )
        ci, _ = camera.get_intrinsics()
        K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
        logger.info(f"本地相机内参: fx={ci.fx:.1f} fy={ci.fy:.1f} cx={ci.ppx:.1f} cy={ci.ppy:.1f}")
        # 测试生成一帧
        test_color, test_depth = camera.get_images()
        logger.info(
            "本地合成摄像头测试成功: color=%s depth=%s",
            test_color.shape, test_depth.shape
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 3. 初始化 GGCNN 模型（仅摄像头识别模式需要）
    # ═════════════════════════════════════════════════════════════════════════
    ggcnn = None
    if mode == RecognitionMode.CAMERA:
        logger.info(f"加载 GGCNN2 模型: {args.model}")
        ggcnn = TorchGGCNN({
            'MODEL_FILE': args.model,
            'OPEN_LOOP_HEIGHT': 0,       # 始终取全局最大值
            'GGCNN_IN_THREAD': False,    # 主线程同步推理
            'DEPTH_CAM_K': K,
        }, Queue(1), Queue(1))
        time.sleep(1)
    else:
        logger.info("手动输入模式：跳过 GGCNN 模型加载")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. 初始化 TaskBuilder
    # ═════════════════════════════════════════════════════════════════════════
    task_config = SimGraspConfig(
        detect_xyz=list(DETECT_XYZ),
        release_xyz=list(RELEASE_XYZ),
        above_z=ABOVE_Z,
        grasping_min_z=GRASPING_MIN_Z,
        move_wait=MOVE_WAIT,
        descend_wait=DESCEND_WAIT,
        suction_wait=SUCTION_WAIT,
        release_wait=RELEASE_WAIT,
    )
    builder = TaskBuilder(task_config)
    logger.info("TaskBuilder 已初始化")

    # ═════════════════════════════════════════════════════════════════════════
    # 4.5 初始化外部坐标输入服务器（可选）
    # ═════════════════════════════════════════════════════════════════════════
    input_server = None
    if mode == RecognitionMode.MANUAL:
        input_server = ExternalInputServer(
            host=args.ext_host, port=args.ext_port
        )
        try:
            input_server.start()
            logger.info(
                "外部坐标输入服务已启动: %s (POST /grasp_target, GET /status)",
                input_server.url
            )
        except OSError as e:
            logger.error(
                "无法启动外部坐标服务器: %s (端口 %d 可能被占用)",
                e, args.ext_port
            )
            logger.error("手动输入模式需要 HTTP 服务器，程序退出。")
            client.close()
            if camera is not None:
                camera.stop()
            sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════════
    # 5. 主循环
    # ═════════════════════════════════════════════════════════════════════════
    cs = min(CAM_HEIGHT, CAM_WIDTH)
    off_r = max(0, CAM_HEIGHT - cs) // 2
    off_c = max(0, CAM_WIDTH - cs) // 2
    cand_buf = []            # 候选目标缓冲
    grasp_count = 0          # 抓取计数
    last_time = 0            # 上次抓取时间
    server_ok = True         # 服务器连接状态

    logger.info("进入主循环 — 放物块即自动识别并发送仿真任务，q/ESC 退出")
    print("\n" + "=" * 60)
    print(f"  仿真模式运行中 [{mode.value.upper()}]")
    print(f"  服务器: {args.server}")
    print("  放物块 → 自动识别 → POST 仿真任务 → 循环")
    print("  q / ESC → 退出")
    print("=" * 60 + "\n")

    while True:
        # ── 获取图像 ──
        try:
            if camera is not None:
                color_image, depth_image = camera.get_images(align=True)
                depth_image = depth_image.astype(np.float32)
            else:
                # 虚拟深度图（用于无相机测试）
                color_image = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
                depth_image = np.ones((CAM_HEIGHT, CAM_WIDTH), dtype=np.float32) * 0.5
        except Exception as e:
            logger.error("获取图像失败: %s", e)
            if args.sim_camera:
                # 仿真摄像头故障：显示错误信息并跳过本帧
                color_image = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
                cv2.putText(color_image, f"CAMERA ERROR: {e}", (10, CAM_HEIGHT//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                depth_image = np.full((CAM_HEIGHT, CAM_WIDTH), math.nan, dtype=np.float32)
                server_ok = False
            else:
                raise

        # ── GGCNN 推理 ──
        eef_pose = get_sim_eef_pose_m()

        # ── 确定抓取目标：按识别模式走单一路径 ──
        goal = None
        stable = False

        if mode == RecognitionMode.MANUAL:
            # ── 手动输入模式：仅从外部 HTTP 服务器取坐标 ──
            external_target = input_server.get_target()

            # 默认占位图（等待 / 未命中）
            grasp_img = np.zeros((cs, cs, 3), dtype=np.uint8)
            cv2.putText(grasp_img, "WAITING", (cs // 3, cs // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(grasp_img, "FOR INPUT", (cs // 3, cs // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if external_target is not None:
                # ── 校验外部坐标 ──
                valid, corrected, reason = validate_grasp_target(
                    external_target, logger
                )
                if not valid:
                    logger.warning(
                        "外部坐标校验失败 — %s — 已丢弃: %s",
                        reason, external_target
                    )
                else:
                    if corrected != external_target:
                        logger.info(
                            "外部坐标已修正: %s → X=%.1f Y=%.1f Z=%.1f Yaw=%.1f",
                            reason, corrected[0], corrected[1],
                            corrected[2], corrected[5]
                        )
                    else:
                        logger.info(
                            "使用外部输入坐标: X=%.1f Y=%.1f Z=%.1f Roll=%.1f Yaw=%.1f",
                            corrected[0], corrected[1],
                            corrected[2], corrected[3],
                            corrected[5]
                        )
                    goal = corrected
                    stable = True
                    cand_buf.clear()
                    # 命中目标：占位图切换为 MANUAL INPUT
                    cv2.putText(grasp_img, "MANUAL", (cs // 3, cs // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(grasp_img, "INPUT", (cs // 3, cs // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 手动模式：color_crop 仅用于显示
            color_crop = color_image[off_r:off_r+cs, off_c:off_c+cs].copy()

        else:
            # ── 摄像头识别模式：GGCNN 推理管线 ──
            grasp_img, result = ggcnn.get_grasp_img(depth_image, K, eef_pose[2])

            color_crop = color_image[off_r:off_r+cs, off_c:off_c+cs].copy()

            if result is not None and result[2] > MIN_RESULT_Z_MM / 1000.0:
                # 相机系 → 基坐标系
                cand = cam_result_to_base(result)
                cand_buf.append(cand)

                if len(cand_buf) > CAND_WINDOW:
                    cand_buf.pop(0)

                if len(cand_buf) >= CAND_WINDOW:
                    pick, cnt = dominant_cluster(cand_buf, CAND_BIN_MM)
                    if cnt >= CAND_MIN_FRAMES:
                        stable = True
                        goal = pick
                    else:
                        goal = cand
                else:
                    goal = cand

                # 绘制抓取点标记
                mp = ggcnn.prev_mp
                cv2.circle(
                    color_crop, (int(mp[1]), int(mp[0])), 6,
                    (0, 255, 0) if stable else (0, 200, 255), -1
                )
            else:
                cand_buf.clear()

        # ── 状态栏 ──
        mode_label = "MANUAL INPUT" if mode == RecognitionMode.MANUAL else "GGCNN"
        if server_ok:
            status = f"{mode_label} | "
        else:
            status = f"{mode_label} [SERVER DOWN] | "
        status += f"{'LOCKED' if stable else ('SEARCHING' if goal is not None else 'NO OBJECT')}"
        status += f" | grasped:{grasp_count}"

        cv2.putText(color_crop, status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(color_crop, f"Server: {args.server}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow(WIN_NAME, get_combined_img(
            color_crop, cv2.resize(grasp_img, (cs, cs))
        ))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # ── 触发抓取 ──
        if (stable and goal is not None
                and in_range(goal)
                and (time.monotonic() - last_time) > COOLDOWN_SEC):

            logger.info(
                "目标已锁定: X=%.1f Y=%.1f Z=%.1f Yaw=%.1f",
                goal[0], goal[1], goal[2], goal[5]
            )

            # 构建任务序列
            sequence = builder.build_pick_and_place(list(goal))

            # 打印序列摘要
            logger.info("生成任务序列 (%d 步):", len(sequence))
            for i, step in enumerate(sequence):
                t = step['type']
                if t == 'move':
                    p = step['params']
                    logger.info(
                        "  %2d. MOVE  (%.0f, %.0f, %.0f) yaw=%.0f  wait=%.1f",
                        i+1, p['x'], p['y'], p['z'], p.get('yaw', 0), step['wait']
                    )
                elif t == 'vacuum':
                    logger.info(
                        "  %2d. VACUUM %s  wait=%.1f",
                        i+1, 'ON' if step['params']['on'] else 'OFF', step['wait']
                    )

            # POST 到仿真服务器
            try:
                resp = client.post_task(sequence)
                logger.info(f"任务已发送 — 响应: {resp}")
                grasp_count += 1
                server_ok = True
            except SimulationClientError as e:
                logger.error(f"发送任务失败: {e}")
                server_ok = False
                print(f"\n[错误] 仿真任务发送失败: {e}")
                print("  请检查仿真服务器状态，程序将继续运行。\n")

            # 重置状态
            last_time = time.monotonic()
            cand_buf.clear()

    # ═════════════════════════════════════════════════════════════════════════
    # 6. 清理
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("正在退出...")
    if input_server is not None:
        input_server.stop()
    if camera is not None:
        camera.stop()
    client.close()
    cv2.destroyAllWindows()
    logger.info(f"结束，共发送 {grasp_count} 个仿真任务")
    print(f"\n结束，共发送 {grasp_count} 个仿真任务")


if __name__ == '__main__':
    main()
