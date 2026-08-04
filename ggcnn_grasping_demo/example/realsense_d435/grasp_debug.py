#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
GGCNN 视觉抓取【调试诊断工具】（不移动机械臂，安全）
-------------------------------------------------------------------
解决两个问题：
  1. 看不到绿点：本工具强制显示 GGCNN 热力图 + 绿色抓取点
  2. 抓不准中心：实时打印/叠加 抓取点像素、相机3D坐标、机械臂基坐标目标点
                 让你对比"AI算的抓取点"和"方块真实位置"的偏差方向，指导调参
-------------------------------------------------------------------
窗口说明（3个区域拼在一起）：
  左：彩色图 + 绿色抓取点 + 抓取方向线
  中：GGCNN 抓取质量热力图（越亮=越适合抓，绿点=最佳抓取点）
  右：深度伪彩色图
控制台会实时打印：像素坐标 / 相机坐标系XYZ / 机械臂基坐标目标 GOAL_POS
-------------------------------------------------------------------
键盘：q/ESC 退出
注意：本工具只【读取】机械臂当前位置用于坐标换算，不会驱动机械臂运动。
     运行前请先在 UFACTORY Studio 里手动把机械臂移动到拍照姿态
     （大致 x=200,y=0,z=380, roll=180,pitch=0,yaw=0），换算才准确。
===================================================================
"""

import os
import sys
import cv2
import time
import numpy as np

# 添加模块搜索路径（和 run_rs_d435_grasp_lite6.py 一致）
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from camera.rs_camera import RealSenseCamera
from grasp.ggcnn_torch import TorchGGCNN
from grasp.helpers.matrix_funcs import euler2mat, convert_pose
from xarm.wrapper import XArmAPI


# ==================== 配置区（和主程序保持一致） ====================
ROBOT_IP = "192.168.2.104"     # 你的机械臂IP
CAM_WIDTH = 640
CAM_HEIGHT = 480

MODEL_FILE = os.path.join(os.path.dirname(__file__), '../../models/ggcnn_epoch_23_cornell')
OPEN_LOOP_HEIGHT = 340

# ★★★ 这两组就是调试重点：手眼外参 ★★★
EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]
EULER_COLOR_TO_DEPTH_OPT = [0, 0, 0, 0, 0, 0]

GRIPPER_Z_MM = 55              # 吸盘长度补偿
CONNECT_ROBOT = True           # True=读取机械臂位置算基坐标；False=只做纯视觉诊断不连机械臂


def get_eef_pose_m(arm):
    """读取机械臂末端位姿（米+弧度），只读，不移动"""
    _, p = arm.get_position(is_radian=True)
    return [p[0]*0.001, p[1]*0.001, p[2]*0.001, p[3], p[4], p[5]]


def main():
    # ---------- 1. 初始化相机 ----------
    camera = RealSenseCamera(width=CAM_WIDTH, height=CAM_HEIGHT)
    color_intrin, _ = camera.get_intrinsics()
    DEPTH_CAM_K = np.array([
        [color_intrin.fx, 0, color_intrin.ppx],
        [0, color_intrin.fy, color_intrin.ppy],
        [0, 0, 1]
    ])
    print("相机内参 fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}".format(
        color_intrin.fx, color_intrin.fy, color_intrin.ppx, color_intrin.ppy))

    # ---------- 2. 初始化GGCNN（不开线程，主线程同步推理） ----------
    from queue import Queue
    ggcnn = TorchGGCNN(
        {'MODEL_FILE': MODEL_FILE, 'OPEN_LOOP_HEIGHT': OPEN_LOOP_HEIGHT,
         'GGCNN_IN_THREAD': False, 'DEPTH_CAM_K': DEPTH_CAM_K},
        Queue(1), Queue(1))
    time.sleep(2)

    # ---------- 3. 连接机械臂（只读位置，不运动） ----------
    arm = None
    if CONNECT_ROBOT:
        arm = XArmAPI(ROBOT_IP, report_type='real')
        time.sleep(0.5)
        print("已连接机械臂（只读位置，不会运动）")

    # 计算裁剪区域（和 ggcnn_torch 内部逻辑一致）
    crop_size = min(min(CAM_HEIGHT, CAM_WIDTH), 500)   # =480
    off_row = max(0, CAM_HEIGHT - crop_size) // 2       # 行偏移=0
    off_col = max(0, CAM_WIDTH - crop_size) // 2        # 列偏移=80

    print("\n开始诊断，把方块放到相机视野中心，观察绿点是否对准方块中心...")
    print("=" * 60)

    frame_cnt = 0
    while True:
        color_image, depth_image = camera.get_images(align=True)
        depth_image = depth_image.astype(np.float32)  # 保证float32，避免numpy2.x下inpaint崩溃(不改官方检测逻辑)

        # 读机械臂当前z（GGCNN开环/闭环判断用），没连就给一个高值
        robot_z = 0.38
        eef_pose = None
        if arm is not None:
            eef_pose = get_eef_pose_m(arm)
            robot_z = eef_pose[2]

        # ---------- GGCNN 推理 ----------
        grasp_img, result = ggcnn.get_grasp_img(depth_image, DEPTH_CAM_K, robot_z)

        # 裁剪彩色图（和GGCNN一致的区域），用来叠加绿点
        color_crop = color_image[off_row:off_row+crop_size, off_col:off_col+crop_size].copy()

        if result is not None:
            x_cam, y_cam, z_cam, ang, width, depth_center = result
            # GGCNN 内部把峰值像素存在 prev_mp（out_size坐标=crop坐标，无缩放）
            mp = ggcnn.prev_mp   # [row, col] in crop
            px, py = int(mp[1]), int(mp[0])   # 在裁剪图里的坐标

            # 在彩色裁剪图上画绿点 + 抓取方向线
            cv2.circle(color_crop, (px, py), 6, (0, 255, 0), -1)
            L = 40
            dx, dy = int(L*np.cos(ang)), int(L*np.sin(ang))
            cv2.line(color_crop, (px-dx, py-dy), (px+dx, py+dy), (0, 255, 0), 2)

            # ---------- 计算机械臂基坐标目标（关键诊断信息） ----------
            base_txt = "base: (need robot)"
            if eef_pose is not None:
                gp = [x_cam, y_cam, z_cam, 0, 0, -1*ang]
                mat = euler2mat(eef_pose) * euler2mat(EULER_EEF_TO_COLOR_OPT) * euler2mat(EULER_COLOR_TO_DEPTH_OPT)
                gp_base = convert_pose(gp, mat)
                gx, gy = gp_base[0]*1000, gp_base[1]*1000
                gz = gp_base[2]*1000 + GRIPPER_Z_MM
                base_txt = "base X={:.1f} Y={:.1f} Z={:.1f} mm".format(gx, gy, gz)

            # 每15帧打印一次，避免刷屏
            frame_cnt += 1
            if frame_cnt % 15 == 0:
                print("[检测到] 像素({},{}) | 相机系 X={:.3f} Y={:.3f} Z={:.3f} m | 角度={:.1f}° | {}".format(
                    px, py, x_cam, y_cam, z_cam, np.degrees(ang), base_txt))

            cv2.putText(color_crop, base_txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(color_crop, "NO GRASP DETECTED", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            frame_cnt += 1
            if frame_cnt % 30 == 0:
                print("[未检测到抓取点] 可能：物体太小/反光/黑色/深度缺失，或不在视野中心")

        # ---------- 深度伪彩色 ----------
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(np.nan_to_num(depth_image*1000), alpha=0.03),
            cv2.COLORMAP_JET)
        depth_crop_vis = depth_vis[off_row:off_row+crop_size, off_col:off_col+crop_size]

        # 统一尺寸拼接显示
        h = 480
        def fit(img):
            return cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h))
        combined = np.hstack([fit(color_crop),
                              fit(cv2.resize(grasp_img, (crop_size, crop_size))),
                              fit(depth_crop_vis)])
        cv2.imshow('DEBUG  [Left:Color+Grasp | Mid:GGCNN Heatmap | Right:Depth]', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    camera.stop()
    if arm is not None:
        arm.disconnect()
    cv2.destroyAllWindows()
    print("诊断结束")


if __name__ == '__main__':
    main()
