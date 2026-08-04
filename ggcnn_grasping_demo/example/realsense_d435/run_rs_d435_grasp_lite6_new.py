#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续全自动 · 两段式开环抓取
==================================================================
去掉官方 RobotGrasp(闭环伺服)，改主循环顺序控制，从根上消除下降漂移。
流程：
  阶段1 观察位识别粗定位  → 得到物块大致base坐标 g1
  阶段2 让相机移到物块正上方 → 停留静置 → 多帧聚类精定位 → 得到 g2
  阶段3 对准 g2 → 直线下降 → 吸取 → 搬运 → 放置 → 回观察位
  → 自动循环
操作：放物块即自动抓，按 q / ESC 退出。
=================================================================="""
import os, sys, cv2, math, time
import numpy as np
from queue import Queue

# 把上两级目录加入搜索路径，才能 import 到 camera / grasp 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from camera.rs_camera import RealSenseCamera        # D435 相机封装
from camera.utils import get_combined_img           # 左右拼图显示工具
from grasp.ggcnn_torch import TorchGGCNN            # GGCNN 抓取点推理
from grasp.helpers.matrix_funcs import euler2mat, convert_pose  # 坐标变换数学
from xarm.wrapper import XArmAPI                     # 机械臂控制

# ==================== 配置区 ====================
ROBOT_IP = "192.168.2.104"                 # 机械臂 IP
WIN_NAME = 'RealSense-D435 AUTO'           # 显示窗口名
CAM_WIDTH, CAM_HEIGHT = 640, 480           # 相机分辨率
MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/epoch_50_cornell')  # GGCNN2 模型权重
EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]  # ★手眼标定：相机相对法兰的位姿(xyz米+rpy弧度)，抓不准调前3个
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]   # 彩色相机↔深度相机偏移(D435已对齐,保持0)
GRASPING_RANGE = [180, 350, -200, 200]     # 抓取安全区 [x_min,x_max,y_min,y_max]，超出不抓
DETECT_XYZ = [200, 0, 380]                 # 观察位(阶段1拍照高度)
RELEASE_XYZ = [225, -89, 83]               # 放置点
GRIPPER_Z_MM = 55                          # 吸盘长度补偿(法兰→吸盘底面)
GRASPING_MIN_Z = 70                        # 允许下降最低Z，防撞桌
MIN_RESULT_Z_MM = 200                      # 相机有效识别最小距离(<0.2m深度失效)
ABOVE_Z = 300                              # 阶段2相机停留高度(别低于~280)
STAY_SEC_AFTER_MOVE = 1.0                  # 到正上方后静置防抖时间(秒)
DESCEND_SPEED = 80                         # 下降速度(低速更稳)
MOVE_SPEED = 150                           # 平移速度
STABLE_FRAMES = 4                          # 观察位连续多少帧稳定才触发(原6)
STABLE_TOL_MM = 5                          # 窗口内抖动峰峰值<5mm 才算稳定
COOLDOWN_SEC = 1.0                         # 两轮抓取最小间隔

FINE_FRAMES = 20                           # 阶段2精定位采集帧数(比阶段1多，延长观察)
FINE_FRAME_INTERVAL = 0.02                 # 每帧间隔，配合帧数延长总采集时间
BIN_MM = 3                                 # 聚类网格(mm)：XY按3mm归类，找最密集格子

SUCTION_WAIT_MS = 800    # 吸取后等待负压建立并检测的时长(ms)，500~1000可调
GRASP_RETRY = 2          # 空抓最多重试次数
TI0_OK_VALUE = 1         # TI0=此值代表"吸住有料"(见下方极性验证，反了就改成0)

# -------- 读机械臂当前末端位姿(mm→m, 姿态用弧度) --------
def get_eef_pose_m(arm):
    _, p = arm.get_position(is_radian=True)
    return [p[0]*0.001, p[1]*0.001, p[2]*0.001, p[3], p[4], p[5]]


# -------- 坐标系转换(重点!!)：GGCNN相机系抓取点 → 机械臂基坐标 --------
def cam_result_to_base(eef_pose, result):
    x, y, z, ang = result[0], result[1], result[2], result[3]
    gp = [x, y, z, 0, 0, -1 * ang]                          # 相机系下的抓取位姿
    # 连乘：基座 ← 末端(实时位姿) ← 相机(手眼标定) ← 深度相机
    mat = euler2mat(eef_pose) * euler2mat(EULER_EEF_TO_COLOR_OPT) * euler2mat(EULER_COLOR_TO_DEPTH_OPT)
    gp_base = convert_pose(gp, mat)                          # 转到机械臂基坐标系
    if gp_base[5] < -np.pi: gp_base[5] += np.pi              # 抓取角度归一化
    elif gp_base[5] > 0:    gp_base[5] -= np.pi
    # 返回 [x_mm, y_mm, z_mm+吸盘补偿, roll=180, pitch=0, yaw度]
    return [gp_base[0]*1000, gp_base[1]*1000, gp_base[2]*1000 + GRIPPER_Z_MM,
            180, 0, math.degrees(gp_base[5] + np.pi/2)]


# -------- 安全区(越界)检查：目标XY是否在允许范围内 --------
def in_range(g):
    return GRASPING_RANGE[0] <= g[0] <= GRASPING_RANGE[1] and GRASPING_RANGE[2] <= g[1] <= GRASPING_RANGE[3]

# -------- 相机偏移换算：算出相机相对法兰在base系的XY偏移 --------
def camera_offset_base_xy(roll, pitch, yaw):
    """给定拍照姿态(弧度)，返回相机相对法兰在base系下的XY偏移(mm)"""
    R = np.array(euler2mat([0, 0, 0, roll, pitch, yaw]))[:3, :3]      # 姿态旋转矩阵
    cam_off_eef = np.array(EULER_EEF_TO_COLOR_OPT[:3]).reshape(3, 1)  # 相机在法兰系位置(m)
    off = np.array(R @ cam_off_eef).flatten() * 1000.0               # 换到base系(mm)
    return off[0], off[1]

# -------- 简单多帧平均识别(备用版，均值,不抗离群) --------
def detect_target(camera, ggcnn, K, arm, frames=6):
    """当前位置连续识别几帧并平均，返回基坐标[x,y,z,180,0,yaw]或None"""
    goals = []
    cs = min(CAM_HEIGHT, CAM_WIDTH)              # 裁剪边长(480)
    off_r = max(0, CAM_HEIGHT - cs) // 2         # 裁剪行偏移
    off_c = max(0, CAM_WIDTH - cs) // 2          # 裁剪列偏移
    for _ in range(frames):
        color, depth = camera.get_images(align=True)
        depth = depth.astype(np.float32)         # numpy2.x兼容
        eef = get_eef_pose_m(arm)                # 关键：用当前实时位姿转换
        gimg, res = ggcnn.get_grasp_img(depth, K, eef[2])
        if res is not None and res[2] > MIN_RESULT_Z_MM / 1000.0:
            goals.append(cam_result_to_base(eef, res))
        cc = color[off_r:off_r+cs, off_c:off_c+cs].copy()
        cv2.imshow(WIN_NAME, get_combined_img(cc, cv2.resize(gimg, (cs, cs))))
        cv2.waitKey(1)
    if len(goals) < frames // 2:                 # 有效帧太少→失败
        return None
    g = np.mean(np.array(goals), axis=0)
    return [g[0], g[1], g[2], 180, 0, g[5]]

# -------- 阶段2精定位：多帧聚类，抗离群(比detect_target更稳) --------
def detect_target_robust(camera, ggcnn, K, arm, frames=FINE_FRAMES, bin_mm=BIN_MM):
    """多帧采集 → 按网格找出现频率最高的簇 → 用该簇内点平均，抗离群"""
    goals = []
    cs = min(CAM_HEIGHT, CAM_WIDTH)
    off_r = max(0, CAM_HEIGHT - cs) // 2
    off_c = max(0, CAM_WIDTH - cs) // 2
    for _ in range(frames):                      # 采集 FINE_FRAMES 帧
        color, depth = camera.get_images(align=True)
        depth = depth.astype(np.float32)
        eef = get_eef_pose_m(arm)
        gimg, res = ggcnn.get_grasp_img(depth, K, eef[2])
        if res is not None and res[2] > MIN_RESULT_Z_MM / 1000.0:
            goals.append(cam_result_to_base(eef, res))
        cc = color[off_r:off_r+cs, off_c:off_c+cs].copy()
        cv2.imshow(WIN_NAME, get_combined_img(cc, cv2.resize(gimg, (cs, cs))))
        cv2.waitKey(1)
        time.sleep(FINE_FRAME_INTERVAL)          # 延长总采集时间

    if len(goals) < max(3, frames // 3):         # 有效帧太少→失败
        return None

    a = np.array(goals)                          # 每行 [x,y,z,180,0,yaw]
    # === 按XY网格找"出现频率最高的簇"，个别跳飞帧会落到别的格子被丢弃 ===
    keys = np.round(a[:, :2] / bin_mm).astype(int)   # XY量化到bin格子
    keys_t = [tuple(k) for k in keys]
    from collections import Counter
    best_key, best_cnt = Counter(keys_t).most_common(1)[0]   # 命中最多的格子
    # 取落在最密集格子(及其相邻±1格)内的所有点
    sel = [i for i, k in enumerate(keys_t)
           if abs(k[0]-best_key[0]) <= 1 and abs(k[1]-best_key[1]) <= 1]
    cluster = a[sel]
    print('[精定位] 有效{}帧, 主簇{}帧'.format(len(goals), len(cluster)))  # 主簇/有效 越高越稳
    g = np.mean(cluster, axis=0)                 # 只对主簇平均
    return [g[0], g[1], g[2], 180, 0, g[5]]

# -------- 两段式抓取主流程 --------
"""def two_stage_grasp(camera, ggcnn, K, arm, g1):
    arm.set_mode(0); arm.set_state(0); time.sleep(0.2)   # 切位置控制模式
    # 阶段2：让【相机】(不是法兰)移到物块正上方 → 停留 → 精定位
    # arm.set_position(x=g1[0], y=g1[1], z=ABOVE_Z, roll=180, pitch=0, yaw=g1[5], speed=MOVE_SPEED, wait=True)
    ox, oy = camera_offset_base_xy(math.pi, 0, 0)        # 拍照姿态(roll=180,yaw=0)下的相机偏移
    arm.set_position(x=g1[0] - ox, y=g1[1] - oy, z=ABOVE_Z,  # 法兰反向退偏移量→相机落到物块正上方
                     roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)

    time.sleep(STAY_SEC_AFTER_MOVE)              # 静置防抖
    g2 = detect_target_robust(camera, ggcnn, K, arm)     # 精定位(g2仍是物块真实base坐标,变换已含相机偏移)
    if g2 is None or not in_range(g2):          # 精定位失败或越界→回观察位放弃本轮
        print('[阶段2] 精定位失败，回观察位')
        arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
        return False
    g2[2] = max(g2[2], GRASPING_MIN_Z)          # 下降高度限位保护
    print('[阶段2] 精定位 X={:.1f} Y={:.1f} Z={:.1f}'.format(g2[0], g2[1], g2[2]))

    # 阶段3：对准g2 → 直线下降抓取 + 负压检测 + 空抓重试
    sucked = False
    for attempt in range(GRASP_RETRY + 1):
        arm.set_position(x=g2[0], y=g2[1], z=ABOVE_Z, roll=180, pitch=0, yaw=g2[5], speed=MOVE_SPEED, wait=True)  # 高位对准
        arm.set_position(z=g2[2], speed=DESCEND_SPEED, wait=True)  # 直线扎下去
        arm.set_vacuum_gripper(on=True)  # 开真空(TO1)
        if check_suction_ok(arm):  # 读TI0判断吸住没
            sucked = True
            print('[吸取] 成功 (第{}次尝试)'.format(attempt + 1))
            break
        print('[吸取] 空抓 (第{}次尝试)，抬升重试'.format(attempt + 1))
        arm.set_vacuum_gripper(on=False)  # 关真空
        arm.set_position(z=ABOVE_Z, speed=MOVE_SPEED, wait=True)  # 抬升后重试
        time.sleep(0.3)

    if not sucked:  # 多次都空抓→放弃本轮
        print('[吸取] 多次空抓，放弃，回观察位')
        arm.set_vacuum_gripper(on=False)
        arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
        arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED,
                         wait=True)
        return False

    # 吸住成功 → 抬升、搬运、放置、回观察位（这部分是你原来的逻辑，保持不变）
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=RELEASE_XYZ[0], y=RELEASE_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED,
                     wait=True)
    arm.set_position(z=RELEASE_XYZ[2], speed=DESCEND_SPEED, wait=True)
    arm.set_vacuum_gripper(on=False);
    time.sleep(0.5)
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED,
                     wait=True)
    print('[完成]')
    return True"""
def two_stage_grasp(camera, ggcnn, K, arm, g1):
    arm.set_mode(0); arm.set_state(0); time.sleep(0.2)
    ox, oy = camera_offset_base_xy(math.pi, 0, 0)   # 拍照姿态下的相机偏移
    aim = g1                                          # 用于算拍照位的定位(初始=粗定位g1)

    # 阶段2+3 作为一个整体重试：每次失败都回精确观察位重新识别
    for attempt in range(GRASP_RETRY + 1):
        # ---- 阶段2：相机移到 aim 正上方 → 停留 → 重新精定位 ----
        arm.set_position(x=aim[0] - ox, y=aim[1] - oy, z=ABOVE_Z,   # 法兰反向退偏移→相机落到物块正上方
                         roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
        time.sleep(STAY_SEC_AFTER_MOVE)              # 静置防抖
        g2 = detect_target_robust(camera, ggcnn, K, arm)   # 精定位
        if g2 is None or not in_range(g2):          # 这次没识别到/越界→回上方重来
            print('[阶段2] 精定位失败 (第{}次)，重试'.format(attempt + 1))
            arm.set_position(z=ABOVE_Z, speed=MOVE_SPEED, wait=True)
            continue
        g2[2] = max(g2[2], GRASPING_MIN_Z)          # 下降限位保护
        aim = g2                                     # 更新aim为最新精定位(下轮拍照位更准)
        print('[阶段2] 精定位 X={:.1f} Y={:.1f} Z={:.1f} (第{}次)'.format(g2[0], g2[1], g2[2], attempt + 1))

        # ---- 阶段3：对准g2 → 直线下降 → 吸取 → 负压检测 ----
        arm.set_position(x=g2[0], y=g2[1], z=ABOVE_Z, roll=180, pitch=0, yaw=g2[5], speed=MOVE_SPEED, wait=True)  # 高位对准
        arm.set_position(z=g2[2], speed=DESCEND_SPEED, wait=True)   # 直线扎下去
        arm.set_vacuum_gripper(on=True)                            # 开真空(TO1)
        if check_suction_ok(arm):                                  # 读TI0判断吸住没
            print('[吸取] 成功 (第{}次尝试)'.format(attempt + 1))
            # 搬运 → 放置 → 回观察位
            arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
            arm.set_position(x=RELEASE_XYZ[0], y=RELEASE_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
            arm.set_position(z=RELEASE_XYZ[2], speed=DESCEND_SPEED, wait=True)
            arm.set_vacuum_gripper(on=False); time.sleep(0.5)
            arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
            arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
            print('[完成]')
            return True

        # ---- 空抓：关真空 → 抬升 → 回精确观察位重新识别(进入下一轮循环) ----
        print('[吸取] 空抓 (第{}次尝试)，回精确观察位重新识别'.format(attempt + 1))
        arm.set_vacuum_gripper(on=False)
        arm.set_position(z=ABOVE_Z, speed=MOVE_SPEED, wait=True)   # 先抬到安全高度，避免拖低Z碰倒物块

    # 所有尝试都失败 → 放弃本轮，回观察位
    print('[吸取] 多次失败，放弃，回观察位')
    arm.set_vacuum_gripper(on=False)
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
    return False

#恢复函数(不退出,仍继续运行)
def recover(arm):
    """检测到错误后：清警→清错→重新使能→回观察位，让程序能继续跑"""
    print('[恢复] 检测到错误 code={}，正在清除并复位...'.format(arm.error_code))
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    arm.set_vacuum_gripper(on=False)                     # 松开吸盘(防止夹着东西)
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2],
                     roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
    print('[恢复] 已回观察位，继续运行')

def check_suction_ok(arm, wait_ms=SUCTION_WAIT_MS):
    """读末端数字输入TI0，判断真空是否吸住物料。吸住返回True"""
    start = time.time()
    while (time.time() - start) < wait_ms / 1000.0:
        ret = arm.get_tgpio_digital(0)          # 读末端TI0，返回(code, value)
        code, val = ret[0], ret[1]
        if isinstance(val, (list, tuple)):      # 有的SDK返回列表，取第0个
            val = val[0]
        if code == 0 and val == TI0_OK_VALUE:   # 负压建立=吸住
            return True
        time.sleep(0.05)
    return False

def main():
    # ---------- 初始化相机 ----------
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    ci, _ = camera.get_intrinsics()                          # 相机内参
    K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
    # ---------- 初始化GGCNN(OPEN_LOOP_HEIGHT=0→始终取全局最大值,配合聚类最稳) ----------
    ggcnn = TorchGGCNN({'MODEL_FILE': MODEL_FILE, 'OPEN_LOOP_HEIGHT': 0,
                        'GGCNN_IN_THREAD': False, 'DEPTH_CAM_K': K}, Queue(1), Queue(1))
    time.sleep(2)
    # ---------- 初始化机械臂并移到观察位 ----------
    arm = XArmAPI(ROBOT_IP, report_type='real')
    arm.motion_enable(True); arm.clean_error(); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
    arm.set_vacuum_gripper(on=False)
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
    print("连续全自动两段式抓取已启动。放物块即自动抓，q 退出。")

    cs = min(CAM_HEIGHT, CAM_WIDTH)
    off_r = max(0, CAM_HEIGHT - cs) // 2
    off_c = max(0, CAM_WIDTH - cs) // 2
    stable_buf = []          # 观察位稳定判定缓冲(存最近几帧XY)
    grasp_count = 0          # 抓取计数
    last_time = 0            # 上次抓取时间(用于冷却)

    # ---------- 主循环：连续检测+自动触发 ----------
    while arm.connected:
        # 错误不退出
        if arm.error_code != 0:
            recover(arm)
            stable_buf.clear()
            last_time = time.monotonic()
            continue

        color_image, depth_image = camera.get_images(align=True)
        depth_image = depth_image.astype(np.float32)         # numpy2.x兼容
        eef = get_eef_pose_m(arm)
        grasp_img, result = ggcnn.get_grasp_img(depth_image, K, eef[2])  # 实时识别
        color_crop = color_image[off_r:off_r+cs, off_c:off_c+cs].copy()

        goal = None; stable = False
        if result is not None and result[2] > MIN_RESULT_Z_MM / 1000.0:
            goal = cam_result_to_base(eef, result)           # 转基坐标
            stable_buf.append(goal[:2])                      # 记录XY
            if len(stable_buf) > STABLE_FRAMES: stable_buf.pop(0)  # 保持窗口长度
            # 窗口满 且 抖动峰峰值<阈值 → 判定稳定
            if len(stable_buf) >= STABLE_FRAMES and np.all(np.ptp(np.array(stable_buf), axis=0) < STABLE_TOL_MM):
                stable = True
                arr = np.array(stable_buf)
                goal = [np.mean(arr[:,0]), np.mean(arr[:,1]), goal[2], 180, 0, goal[5]]  # 用窗口均值
            mp = ggcnn.prev_mp                               # 抓取点像素
            cv2.circle(color_crop, (int(mp[1]), int(mp[0])), 6, (0,255,0) if stable else (0,200,255), -1)  # 绿=锁定/橙=搜索
        else:
            stable_buf.clear()                               # 没识别到→清空缓冲

        # 状态栏显示
        status = "LOCKED" if stable else ("SEARCHING" if goal is not None else "NO OBJECT")
        cv2.putText(color_crop, "{} grasped:{}".format(status, grasp_count), (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.imshow(WIN_NAME, get_combined_img(color_crop, cv2.resize(grasp_img, (cs, cs))))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break               # q/ESC退出

        # 触发条件：稳定 + 在安全区 + 过冷却时间 → 执行一次两段式抓取
        if stable and in_range(goal) and (time.monotonic() - last_time) > COOLDOWN_SEC:
            if two_stage_grasp(camera, ggcnn, K, arm, list(goal)): grasp_count += 1
            last_time = time.monotonic(); stable_buf.clear()

    # ---------- 退出清理 ----------
    camera.stop(); arm.disconnect(); cv2.destroyAllWindows()
    print("结束，共抓取 {} 次".format(grasp_count))


if __name__ == '__main__':
    main()