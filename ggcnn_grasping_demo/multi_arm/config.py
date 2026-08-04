#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config — 多臂协同系统配置
==========================
定义 ArmConfig / SystemConfig 数据类、ArmState / ZoneState 枚举、
JSON 配置文件加载函数。
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 状态枚举
# ═══════════════════════════════════════════════════════════════════════════

class ArmState(Enum):
    """机械臂运行状态"""
    IDLE       = "IDLE"        # 在观察位，搜索目标中
    LOCKED     = "LOCKED"      # 已稳定锁定抓取目标
    MOVING     = "MOVING"      # 平移运动中
    GRASPING   = "GRASPING"    # 阶段2/3：精定位 + 下降抓取
    PLACING    = "PLACING"     # 搬运到释放点
    RECOVERING = "RECOVERING"  # 错误恢复中
    STOPPED    = "STOPPED"     # 严重故障，需人工介入
    DISCONNECTED = "DISCONNECTED"  # 网络断开


class ZoneState(Enum):
    """区域占用状态"""
    FREE   = "FREE"    # 空闲
    OWNED  = "OWNED"   # 已被某臂持有
    HAZARD = "HAZARD"  # 危险区（L2/L3 故障残留，需人工清除）


# ═══════════════════════════════════════════════════════════════════════════
# 配置数据类
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArmConfig:
    """单臂完整配置 — 与原始脚本中的全局常量一一对应"""

    # ── 标识与连接 ──
    arm_id: int                             # 臂编号 (0, 1, 2)
    name: str                               # 臂名称 "Arm-Left" / "Arm-Center" / "Arm-Right"
    robot_ip: str                           # 机械臂 IP 地址

    # ── 手眼标定 ──
    euler_eef_to_color_opt: List[float]     # 相机在法兰系位姿 (xyz_m + rpy_rad)
    euler_color_to_depth_opt: List[float]   # 深度相机在彩色相机系位姿 (D435已对齐，全0)

    # ── 工作区 ──
    workspace_zone: List[float]             # [x_min, x_max, y_min, y_max] 独占区 (mm)
    coordination_zones: List[List[float]] = field(default_factory=list)  # 该臂可能访问的协调区列表

    # ── 关键位姿 ──
    detect_xyz: List[float] = field(default_factory=lambda: [200, 0, 380])    # 观察位
    release_xyz: List[float] = field(default_factory=lambda: [225, -89, 83])  # 释放/放置点

    # ── 相机参数 ──
    cam_width: int = 640
    cam_height: int = 480

    # ── 模型路径 ──
    model_file: str = ""                    # GGCNN2 权重文件路径 (空则用默认)

    # ── 抓取参数 ──
    gripper_z_mm: int = 55                  # 吸盘长度补偿 (法兰→吸盘底面)
    grasping_min_z: int = 70                # 允许下降最低 Z，防撞桌
    above_z: int = 300                      # 阶段2 相机悬停高度
    min_result_z_mm: int = 200              # 相机有效识别最小距离 (<200mm深度失效)

    # ── 运动速度 ──
    move_speed: int = 150                   # 平移速度 (mm/s)
    descend_speed: int = 80                 # 下降速度 (mm/s)

    # ── 观测稳定性参数 ──
    cand_window: int = 12                   # 候选缓冲窗口帧数
    cand_bin_mm: int = 8                    # 物块聚类网格 (mm)
    cand_min_frames: int = 6                # 主簇最少帧数（触发 LOCKED）
    fine_frames: int = 20                   # 精定位采集帧数
    fine_frame_interval: float = 0.02       # 精定位帧间隔 (秒)
    bin_mm: int = 3                         # 精定位聚类网格 (mm)
    stay_sec_after_move: float = 1.0        # 到位后静置防抖时间 (秒)

    # ── 吸取参数 ──
    suction_wait_ms: int = 800              # 吸取后等待负压建立 (ms)
    grasp_retry: int = 2                    # 空抓最多重试次数
    ti0_ok_value: int = 1                   # TI0=此值代表"吸住有料"
    cooldown_sec: float = 1.0               # 两轮抓取最小间隔 (秒)

    def __post_init__(self):
        """初始化后处理：设置默认模型路径"""
        if not self.model_file:
            self.model_file = os.path.join(
                os.path.dirname(__file__),
                '..', 'models', 'epoch_50_cornell'
            )


@dataclass
class SystemConfig:
    """多臂系统全局配置"""

    arms: List[ArmConfig]

    # ── 安全参数 ──
    safety_height: float = 350.0            # 跨区安全高度 (mm)，低于此高度不可经过他人区域
    safety_radius_mm: float = 100.0         # 两臂最小安全间距 (mm)
    lease_duration_s: float = 30.0          # 区域锁租约时长 (秒)
    lease_renew_interval_s: float = 5.0     # 续约间隔 (秒)
    zone_request_timeout_s: float = 5.0     # 协调区请求超时 (秒)

    # ── 全局调度 ──
    global_cooldown_ms: int = 500           # 两臂抓取最小间隔 (ms)

    # ── 显示设置 ──
    display_layout: str = "horizontal"      # 布局模式: "horizontal" | "vertical"
    win_name: str = "3-Arm Collaborative Grasping"  # 窗口标题

    # ── 监控参数 ──
    safety_monitor_hz: float = 10.0         # 安全距离检查频率 (Hz)
    position_stale_s: float = 3.0           # 位置数据过期时间 (秒)，判定 L2 断连
    ee_stuck_s: float = 10.0                # EE 静止不动时间 (秒)，判定 L3 卡死

    def get_arm(self, arm_id: int) -> Optional[ArmConfig]:
        """按 arm_id 查找对应配置"""
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        return None


# ═══════════════════════════════════════════════════════════════════════════
# JSON 配置加载
# ═══════════════════════════════════════════════════════════════════════════

def load_config(json_path: str) -> SystemConfig:
    """从 JSON 文件加载系统配置。

    参数:
        json_path: config_3arms.json 的路径

    返回:
        SystemConfig 实例
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    arms = []
    for a in raw.get('arms', []):
        arms.append(ArmConfig(
            arm_id=a['arm_id'],
            name=a.get('name', f"Arm-{a['arm_id']}"),
            robot_ip=a['robot_ip'],
            euler_eef_to_color_opt=a['euler_eef_to_color_opt'],
            euler_color_to_depth_opt=a.get('euler_color_to_depth_opt',
                                           [0, 0, 0, 0, 0, 0]),
            workspace_zone=a['workspace_zone'],
            coordination_zones=a.get('coordination_zones', []),
            detect_xyz=a.get('detect_xyz', [200, 0, 380]),
            release_xyz=a.get('release_xyz', [225, -89, 83]),
            cam_width=a.get('cam_width', 640),
            cam_height=a.get('cam_height', 480),
            model_file=a.get('model_file', ''),
            gripper_z_mm=a.get('gripper_z_mm', 55),
            grasping_min_z=a.get('grasping_min_z', 70),
            above_z=a.get('above_z', 300),
            min_result_z_mm=a.get('min_result_z_mm', 200),
            move_speed=a.get('move_speed', 150),
            descend_speed=a.get('descend_speed', 80),
            cand_window=a.get('cand_window', 12),
            cand_bin_mm=a.get('cand_bin_mm', 8),
            cand_min_frames=a.get('cand_min_frames', 6),
            fine_frames=a.get('fine_frames', 20),
            fine_frame_interval=a.get('fine_frame_interval', 0.02),
            bin_mm=a.get('bin_mm', 3),
            stay_sec_after_move=a.get('stay_sec_after_move', 1.0),
            suction_wait_ms=a.get('suction_wait_ms', 800),
            grasp_retry=a.get('grasp_retry', 2),
            ti0_ok_value=a.get('ti0_ok_value', 1),
            cooldown_sec=a.get('cooldown_sec', 1.0),
        ))

    return SystemConfig(
        arms=arms,
        safety_height=raw.get('safety_height', 350.0),
        safety_radius_mm=raw.get('safety_radius_mm', 100.0),
        lease_duration_s=raw.get('lease_duration_s', 30.0),
        lease_renew_interval_s=raw.get('lease_renew_interval_s', 5.0),
        zone_request_timeout_s=raw.get('zone_request_timeout_s', 5.0),
        global_cooldown_ms=raw.get('global_cooldown_ms', 500),
        display_layout=raw.get('display_layout', 'horizontal'),
        win_name=raw.get('win_name', '3-Arm Collaborative Grasping'),
        safety_monitor_hz=raw.get('safety_monitor_hz', 10.0),
        position_stale_s=raw.get('position_stale_s', 3.0),
        ee_stuck_s=raw.get('ee_stuck_s', 10.0),
    )


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 构造最小配置测试
    cfg = ArmConfig(
        arm_id=0, name="Test",
        robot_ip="192.168.1.100",
        euler_eef_to_color_opt=[0.067, -0.031, 0.022, -0.004, -0.008, 1.590],
        euler_color_to_depth_opt=[0, 0, 0, 0, 0, 0],
        workspace_zone=[100, 500, -450, -100],
    )
    print(f"ArmConfig: {cfg.name} @ {cfg.robot_ip}")
    print(f"  独占区: {cfg.workspace_zone}")
    print(f"  观察位: {cfg.detect_xyz}")
    sys_cfg = SystemConfig(arms=[cfg])
    print(f"SystemConfig: {len(sys_cfg.arms)} arm(s), safety_height={sys_cfg.safety_height}mm")
    print("PASS: config.py OK")
