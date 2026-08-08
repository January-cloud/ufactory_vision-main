#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_3arm — 三臂协同仿真包
===========================
仿真平台上的三臂协同抓取系统，结构对齐真机 multi_arm 系统：

  真机 multi_arm               仿真 sim_3arm
  ----------------------       ----------------------
  XArmAPI + RealSense D435     SimulationClient + BuiltinCamera/SimCamera
  MultiArmCoordinator          SimCoordinator (包装复用)
  ArmController (线程)          SimArmController (线程)
  MultiArmVisualizer           SimVisualizer (含全局摄像头)
  run_3arm_grasp.py            run_sim_3arm.py

本包新增一个真机没有的组件：全局俯瞰摄像头 (GlobalCamera)，覆盖全部
三臂工作区，用于整体场景监控。
"""

import os
import sys

from simulation.sim_3arm.sim_arm_controller import SimArmController
from simulation.sim_3arm.sim_coordinator import SimSystemConfig, SimCoordinator, load_sim_config
from simulation.sim_3arm.sim_visualizer import SimVisualizer
from simulation.sim_3arm.task_recorder import TaskRecorder

# 将 simulation/ 目录和 ggcnn_grasping_demo/ 目录加入 sys.path，
# 保证直接运行 sim_3arm/ 下脚本时能 import 到：
#   simulation_client / task_builder / builtin_camera / global_camera
#   multi_arm.*
_sim_dir = os.path.dirname(os.path.abspath(__file__))       # .../simulation/sim_3arm
_simulation_dir = os.path.dirname(_sim_dir)                 # .../simulation
_demo_dir = os.path.dirname(_simulation_dir)                # .../ggcnn_grasping_demo
for _p in (_simulation_dir, _demo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = [
    'SimSystemConfig',
    'load_sim_config',
    'SimCoordinator',
    'SimArmController',
    'SimVisualizer',
    'TaskRecorder',
]
