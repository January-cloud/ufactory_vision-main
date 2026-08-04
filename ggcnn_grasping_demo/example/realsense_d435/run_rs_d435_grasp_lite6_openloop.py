#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
D435 + Lite6 真空吸盘  【开环视觉抓取】版本（针对静止物体优化）
-------------------------------------------------------------------
和官方闭环版本的区别：
  官方(闭环)：移动到目标上方 → 边下降边根据新识别微调XY(伺服模式7) → 抓
             缺点：下降中相机变近、AI预测跳变，导致XY漂移抓偏
  本版(开环)：在高位识别一次 → 锁定目标XY/角度 → 直线扎下去 → 抓
             优点：静止物体不再漂移，抓取稳定命中中心
-------------------------------------------------------------------
流程：
  1. 机械臂移动到拍照点 DETECT_XYZ 静止
  2. 持续识别，窗口显示绿点（对准物体中心即可）
  3. 按 g 键 → 锁定当前目标，执行一次完整开环抓取（直线下降，不再调整）
  4. 抓完自动回拍照点，可继续按 g 抓下一个
  5. 按 q / ESC 退出
  （如需全自动连续抓取，把 AUTO_GRASP 改成 True）
-------------------------------------------------------------------
安全：低速运行，手放急停旁；首次务必用 g 键手动触发，确认无误再开 AUTO
===================================================================
"""

import os
import sys
import cv2
import math
import time
import numpy as np
from queue import Queue

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from camera.rs_camera import RealSenseCamera
from grasp.ggcnn_torch import TorchGGCNN
from grasp.helpers.matrix_funcs import euler2mat, convert_pose
from xarm.wrapper import XArmAPI


# ==================== 配置区（与主程序保持一致） ====================
ROBOT_IP = "192.168.2.104"
CAM_WIDTH, CAM_HEIGHT = 640, 480
MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/ggcnn_epoch_23_cornell')
OPEN_LOOP_HEIGHT = 340

EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

GRASPING_RANGE = [180, 350, -200, 200]   # [x_min, x_max, y_min, y_max]
DETECT_XYZ = [200, 0, 380]               # 拍照点
RELEASE_XYZ = [200, 200, 200]            # 放置点
GRIPPER_Z_MM = 55                        # 吸盘长度补偿(和Studio里TCP要二选一，别重复设)
GRASPING_MIN_Z = 70                      # 允许下降最低Z，防撞桌
MIN_RESULT_Z_MM = 200                    # 相机有效识别最小距离(m内)

DESCEND_SPEED = 80                        # 下降速度(低速更稳更安全)
MOVE_SPEED = 150                          # 平移速度
AUTO_GRASP = False                        # False=按g触发；True=自动连续抓取
STABLE_FRAMES = 5                         # 锁定前平均多少帧，抑制抖动


def get_eef_pose_m(arm):
    """读末端位姿（米+弧度）"""
    _, p = arm.get_position(is_radian=True)
    return [p[0]*0.001, p[1]*0.001, p[2]*0.001, p[3], p[4], p[5]]


def cam_result_to_base(eef_pose, result):
    """把GGCNN相机坐标系抓取结果 → 机械臂基坐标系目标位姿(mm/deg)
       完全复用官方 robot_grasp_lite6.grasp() 的坐标变换数学"""
    x, y, z, ang = result[0], result[1], result[2], result[3]
    gp = [x, y, z, 0, 0, -1 * ang]                 # 相机系下的抓取位姿(米)
    mat = euler2mat(eef_pose) * euler2mat(EULER_EEF_TO_COLOR_OPT) * euler2mat(EULER_COLOR_TO_DEPTH_OPT)
    gp_base = convert_pose(gp, mat)                 # 转到机械臂基坐标系
    # 角度归一化（同官方）
    if gp_base[5] < -np.pi:
        gp_base[5] += np.pi
    elif gp_base[5] > 0:
        gp_base[5] -= np.pi
    yaw_deg = math.degrees(gp_base[5] + np.pi / 2)
    return [gp_base[0]*1000, gp_base[1]*1000, gp_base[2]*1000 + GRIPPER_Z_MM, 180, 0, yaw_deg]


def do_open_loop_grasp(arm, goal):
    """开环抓取：直线下降，全程不再读取/更新视觉"""
    # 边界与最低高度保护
    goal[2] = max(goal[2], GRASPING_MIN_Z)
    if goal[0] < GRASPING_RANGE[0] or goal[0] > GRASPING_RANGE[1] \
       or goal[1] < GRASPING_RANGE[2] or goal[1] > GRASPING_RANGE[3]:
        print('[跳过] 目标超出安全范围: {}'.format([round(v, 1) for v in goal]))
        return False

    print('[开环抓取] 锁定目标 X={:.1f} Y={:.1f} Z={:.1f} yaw={:.1f}'.format(
        goal[0], goal[1], goal[2], goal[5]))

    # 位置控制模式，逐步执行，全程 wait=True，中间不读视觉
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)

    # 1. 先在高位对准目标XY和角度（Z保持拍照高度）
    arm.set_position(x=goal[0], y=goal[1], z=DETECT_XYZ[2],
                     roll=180, pitch=0, yaw=goal[5], speed=MOVE_SPEED, wait=True)
    # 2. 直线扎下去到抓取高度（关键：只动Z，XY锁死，不再微调）
    arm.set_position(z=goal[2], speed=DESCEND_SPEED, wait=True)
    # 3. 开真空吸盘
    arm.set_vacuum_gripper(on=True)
    time.sleep(0.5)
    # 4. 抬升
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    # 5. 移动到放置点上方 → 下降 → 松开
    arm.set_position(x=RELEASE_XYZ[0], y=RELEASE_XYZ[1], z=DETECT_XYZ[2],
                     roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
    arm.set_position(z=RELEASE_XYZ[2], speed=DESCEND_SPEED, wait=True)
    arm.set_vacuum_gripper(on=False)
    time.sleep(0.5)
    # 6. 回拍照点
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2],
                     roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
    print('[完成] 已放置并回到拍照点')
    return True


def main():
    # ---------- 相机 ----------
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    color_intrin, _ = camera.get_intrinsics()
    DEPTH_CAM_K = np.array([
        [color_intrin.fx, 0, color_intrin.ppx],
        [0, color_intrin.fy, color_intrin.ppy],
        [0, 0, 1]])

    # ---------- GGCNN ----------
    ggcnn = TorchGGCNN(
        {'MODEL_FILE': MODEL_FILE, 'OPEN_LOOP_HEIGHT': OPEN_LOOP_HEIGHT,
         'GGCNN_IN_THREAD': False, 'DEPTH_CAM_K': DEPTH_CAM_K},
        Queue(1), Queue(1))
    time.sleep(2)

    # ---------- 机械臂初始化，移动到拍照点 ----------
    arm = XArmAPI(ROBOT_IP, report_type='real')
    arm.motion_enable(True)
    arm.clean_error()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    arm.set_vacuum_gripper(on=False)
    arm.set_position(z=DETECT_XYZ[2], speed=MOVE_SPEED, wait=True)
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2],
                     roll=180, pitch=0, yaw=0, speed=MOVE_SPEED, wait=True)
    print("机械臂已到拍照点。把方块放视野中心，绿点对准后按 g 抓取，q 退出。")

    crop_size = min(min(CAM_HEIGHT, CAM_WIDTH), 500)
    off_row = max(0, CAM_HEIGHT - crop_size) // 2
    off_col = max(0, CAM_WIDTH - crop_size) // 2

    goal_buffer = []   # 用于多帧平均，锁定前抑制抖动

    while arm.connected and arm.error_code == 0:
        color_image, depth_image = camera.get_images(align=True)
        depth_image = depth_image.astype(np.float32)  # 保证float32，避免numpy2.x下inpaint崩溃(不改官方检测逻辑)
        eef_pose = get_eef_pose_m(arm)
        # 只在高位识别（保证是稳定的全局最大值），robot_z传当前高度
        grasp_img, result = ggcnn.get_grasp_img(depth_image, DEPTH_CAM_K, eef_pose[2])

        color_crop = color_image[off_row:off_row+crop_size, off_col:off_col+crop_size].copy()

        goal = None
        if result is not None and result[2] > MIN_RESULT_Z_MM / 1000.0:
            goal = cam_result_to_base(eef_pose, result)
            # 多帧滑动平均，抑制抖动
            goal_buffer.append(goal[:3] + [goal[5]])
            if len(goal_buffer) > STABLE_FRAMES:
                goal_buffer.pop(0)
            avg = np.mean(goal_buffer, axis=0)
            goal = [avg[0], avg[1], avg[2], 180, 0, avg[3]]

            # 可视化绿点
            mp = ggcnn.prev_mp
            px, py = int(mp[1]), int(mp[0])
            cv2.circle(color_crop, (px, py), 6, (0, 255, 0), -1)
            ang = result[3]
            dx, dy = int(40*np.cos(ang)), int(40*np.sin(ang))
            cv2.line(color_crop, (px-dx, py-dy), (px+dx, py+dy), (0, 255, 0), 2)
            cv2.putText(color_crop, "target X={:.0f} Y={:.0f} Z={:.0f}".format(goal[0], goal[1], goal[2]),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            goal_buffer.clear()
            cv2.putText(color_crop, "NO GRASP", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        heat = cv2.resize(grasp_img, (crop_size, crop_size))
        combined = np.hstack([color_crop, heat])
        cv2.imshow('OpenLoop Grasp [Left:Color+Point | Right:GGCNN Heatmap]  (g=grasp q=quit)', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        # 触发抓取：手动按g，或AUTO模式下目标稳定即抓
        trigger = (key == ord('g')) or (AUTO_GRASP and goal is not None and len(goal_buffer) >= STABLE_FRAMES)
        if trigger and goal is not None:
            do_open_loop_grasp(arm, list(goal))
            goal_buffer.clear()
            time.sleep(0.5)

    camera.stop()
    arm.disconnect()
    cv2.destroyAllWindows()
    print("程序结束")


if __name__ == '__main__':
    main()
