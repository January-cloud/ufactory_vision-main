# -*- coding: utf-8 -*-
"""
多臂协同抓取系统
================
三台 xArm Lite 6 + RealSense D435 + GGCNN2 视觉引导协同抓取，
带碰撞避免功能。

模块说明:
    config              - 数据类配置 + JSON 加载器
    dominant_cluster    - 基于网格的候选抓取点聚类
    collision_avoidance - 区域边界计算工具
    coordinator         - 中心协调器 (状态、区域、安全)
    arm_controller      - 单臂控制线程，封装完整抓取管线
    visualizer          - 多视图组合 OpenCV 显示窗口
    run_3arm_grasp      - 主启动入口
"""

# 无条件导入 (纯 Python，无硬件依赖)
from .config import ArmConfig, SystemConfig, ArmState, ZoneState, load_config
from .dominant_cluster import dominant_cluster

# 延迟导入 — 依赖硬件 (pyrealsense2, xarm, torch) 的模块在需要时才加载
# 用户应直接 import:
#   from multi_arm.coordinator import MultiArmCoordinator
#   from multi_arm.arm_controller import ArmController
#   from multi_arm.visualizer import MultiArmVisualizer

__all__ = [
    'ArmConfig', 'SystemConfig', 'ArmState', 'ZoneState', 'load_config',
    'dominant_cluster',
]
