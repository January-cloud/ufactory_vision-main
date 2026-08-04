#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
D435 深度相机学习测试脚本（新手入门专用，中文详细注释）
-------------------------------------------------------------------
功能：
  1. 同时打开 D435 的【彩色图】和【深度伪彩色图】两个窗口
  2. 鼠标在彩色图上移动，实时显示该点的真实距离（单位：米/毫米）
  3. 鼠标左键点击，把该点的像素坐标 + 距离 + 相机坐标系3D坐标 打印出来
  4. 左上角实时显示相机内参 fx, fy, cx, cy
  5. 按 s 键：保存当前彩色图和深度图到本地；按 q / ESC 退出
-------------------------------------------------------------------
目的：帮你直观理解「深度相机到底给了什么数据」，为后续视觉抓取打基础
运行：直接 python d435_test_learn.py（不需要连接机械臂，最安全）
===================================================================
"""

import cv2
import time
import numpy as np
import pyrealsense2 as rs   # Intel RealSense 官方驱动库


# ==================== 全局变量：存放鼠标当前指向的点 ====================
mouse_x, mouse_y = 320, 240   # 鼠标当前坐标，默认画面中心
click_info = ""               # 鼠标点击后要显示的文字信息


# ==================== 鼠标回调函数 ====================
# 只要鼠标在窗口里移动/点击，OpenCV 就会自动调用这个函数
def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, click_info
    mouse_x, mouse_y = x, y   # 实时记录鼠标位置

    # 鼠标左键点击时，记录该点信息（真正取值在主循环里做）
    if event == cv2.EVENT_LBUTTONDOWN:
        click_info = "CLICK"  # 标记发生了点击，主循环会处理


def main():
    global click_info

    # ==================== 1. 初始化 D435 相机 ====================
    pipeline = rs.pipeline()          # 创建数据管道
    config = rs.config()              # 创建配置对象

    # 开启【深度流】：640x480，16位深度格式，30帧
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # 开启【彩色流】：640x480，BGR8彩色格式，30帧
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)  # 启动相机

    # 创建对齐对象：把深度图对齐到彩色图坐标系
    # 作用：让彩色图和深度图的每个像素一一对应（同一像素点=同一物理位置）
    align = rs.align(rs.stream.color)

    # ==================== 2. 获取相机内参 ====================
    # 内参 = 相机的固有属性，用来把「像素坐标」换算成「真实3D坐标」
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    fx, fy = intr.fx, intr.fy   # 焦距（像素单位）
    cx, cy = intr.ppx, intr.ppy # 光心（画面中心像素坐标）
    print("=" * 50)
    print("D435 相机内参：")
    print("  fx = {:.2f}  fy = {:.2f}".format(fx, fy))
    print("  cx = {:.2f}  cy = {:.2f}".format(cx, cy))
    print("=" * 50)
    print("操作说明：鼠标移动看距离 | 左键点击打印3D坐标 | s保存 | q退出")

    # 创建窗口并绑定鼠标回调
    cv2.namedWindow('COLOR')
    cv2.setMouseCallback('COLOR', on_mouse)

    try:
        # ==================== 3. 主循环：持续读取画面 ====================
        while True:
            # 获取一帧数据并对齐
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()   # 深度帧
            color_frame = aligned.get_color_frame()   # 彩色帧
            if not depth_frame or not color_frame:
                continue

            # 转成 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())   # 彩色图 (480,640,3)
            depth_image = np.asanyarray(depth_frame.get_data())   # 深度图 (480,640) 单位毫米，16位整数

            # ---------- 深度图可视化：转成彩虹伪彩色，方便肉眼看 ----------
            # 原始深度图是16位灰度，人眼看不清，用applyColorMap变成彩色
            # 近的地方红色，远的地方蓝色
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # ---------- 读取鼠标所在点的距离 ----------
            # depth_frame.get_distance(x, y) 直接返回该像素的真实距离（单位：米）
            mx = min(max(mouse_x, 0), 639)
            my = min(max(mouse_y, 0), 479)
            dist_m = depth_frame.get_distance(mx, my)  # 距离，单位米

            # ---------- 在彩色图上画十字准星 + 显示距离 ----------
            cv2.drawMarker(color_image, (mx, my), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
            dist_text = "dist: {:.3f} m ({:.0f} mm)".format(dist_m, dist_m * 1000)
            cv2.putText(color_image, dist_text, (mx + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---------- 左上角显示内参 ----------
            cv2.putText(color_image, "fx={:.0f} fy={:.0f} cx={:.0f} cy={:.0f}".format(fx, fy, cx, cy),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # ---------- 处理鼠标点击：打印该点3D坐标 ----------
            if click_info == "CLICK":
                click_info = ""
                if dist_m > 0:
                    # rs.rs2_deproject_pixel_to_point：像素坐标 + 距离 → 相机坐标系下的3D点(米)
                    point_3d = rs.rs2_deproject_pixel_to_point(intr, [mx, my], dist_m)
                    print("-" * 50)
                    print("点击像素: ({}, {})".format(mx, my))
                    print("该点距离: {:.3f} 米 = {:.1f} 毫米".format(dist_m, dist_m * 1000))
                    print("相机坐标系3D坐标(米): X={:.3f}  Y={:.3f}  Z={:.3f}".format(
                        point_3d[0], point_3d[1], point_3d[2]))
                    print("  含义：X右为正 / Y下为正 / Z前方(距离)为正")
                else:
                    print("该点无有效深度（可能太近/太远/反光/黑色物体）")

            # ---------- 显示两个窗口 ----------
            cv2.imshow('COLOR', color_image)        # 彩色图 + 准星 + 距离
            cv2.imshow('DEPTH', depth_colormap)     # 深度伪彩色图

            # ---------- 键盘控制 ----------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:        # q 或 ESC 退出
                break
            elif key == ord('s'):                   # s 保存图片
                ts = time.strftime('%H%M%S')
                cv2.imwrite('d435_color_{}.png'.format(ts), color_image)
                cv2.imwrite('d435_depth_{}.png'.format(ts), depth_colormap)
                print("已保存: d435_color_{}.png / d435_depth_{}.png".format(ts, ts))

    finally:
        # ==================== 4. 退出清理 ====================
        pipeline.stop()          # 关闭相机
        cv2.destroyAllWindows()  # 关闭所有窗口
        print("相机已关闭，程序退出")


if __name__ == '__main__':
    main()
