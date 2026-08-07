#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation — 仿真接口模块
=========================
提供基于 HTTP 的仿真平台通信接口，将 GGCNN 抓取推理结果转换为
仿真平台可执行的任务序列（move / vacuum 动作），通过 POST 发送到仿真服务器。

与物理机械臂模式的主要区别:
  - 无 xArm SDK 依赖：用 HTTP POST 替代 arm.set_position() / arm.set_vacuum_gripper()
  - 单阶段抓取：相机固定不动（无 Phase 2 精定位重拍），从观察位直接抓取
  - 无实时反馈：不轮询位置/TI0/错误码，用序列中的 wait 字段模拟动作完成时间
  - 零硬件依赖：默认使用本地合成摄像头，无需 RealSense D435 或 pyrealsense2

摄像头选项:
  - BuiltinCamera (默认): 本地合成 RGB-D 图像，无需任何外部依赖
  - SimCamera: 从 HTTP 仿真服务器获取图像
  - --no-camera: 静态虚拟图像 (测试用)

外部输入:
  - ExternalInputServer: HTTP 服务器接收外部抓取坐标 (POST /grasp_target)

模块:
    SimulationClient      - 仿真服务器 HTTP 客户端
    TaskBuilder           - 抓取目标 → 动作序列转换器
    SimGraspConfig        - 仿真抓取配置数据类
    SimCamera             - 仿真摄像头适配器（HTTP → numpy）
    BuiltinCamera         - 本地合成摄像头（零依赖）
    GlobalCamera          - 全局俯瞰摄像头（覆盖三臂工作区）
    ExternalInputServer   - 外部坐标输入 HTTP 服务器

脚本:
    run_simulation.py - 单臂仿真模式主入口
    sim_3arm/         - 三臂协同仿真包 (run_sim_3arm.py 主入口)
"""

from .simulation_client import SimulationClient, SimulationClientError
from .task_builder import TaskBuilder, SimGraspConfig
from .sim_camera import SimCamera
from .builtin_camera import BuiltinCamera
from .global_camera import GlobalCamera
from .external_input import ExternalInputServer

__all__ = [
    'SimulationClient',
    'SimulationClientError',
    'TaskBuilder',
    'SimGraspConfig',
    'SimCamera',
    'BuiltinCamera',
    'GlobalCamera',
    'ExternalInputServer',
]
