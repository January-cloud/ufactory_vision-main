#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_coordinator — 三臂仿真协调器 + 仿真配置加载
=================================================
1. 定义仿真专用配置数据类 (SimSystemConfig / GlobalCameraConfig)
2. 提供 load_sim_config() 加载 config_sim_3arms.json
3. SimCoordinator：薄封装真机 MultiArmCoordinator，复用其区域锁、
   安全监控、死锁检测等全部逻辑，并补充仿真专用能力：
     - 共享 SimulationClient 引用
     - 全局抓取冷却 + 任务发送串行化 (同一时刻仅一个臂 POST /task)
"""

import os
import sys
import json
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── 包内路径引导：保证能 import 到 simulation 与 multi_arm ──
_sim_3arm_dir = os.path.dirname(os.path.abspath(__file__))
_simulation_dir = os.path.dirname(_sim_3arm_dir)
_demo_dir = os.path.dirname(_simulation_dir)
for _p in (_simulation_dir, _demo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulation.simulation_client import SimulationClient, SimulationClientError
from multi_arm.config import SystemConfig, ArmConfig, load_config
from multi_arm.coordinator import MultiArmCoordinator
from simulation.sim_3arm.task_recorder import TaskRecorder


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真专用配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GlobalCameraConfig:
    """全局俯瞰摄像头配置。"""
    width: int = 640
    height: int = 480
    table_depth_m: float = 0.50
    noise_std_m: float = 0.0015
    jitter_px: float = 3.0
    objects: List[Dict] = field(default_factory=list)


@dataclass
class SimSystemConfig:
    """三臂仿真系统配置。

    包装真机 SystemConfig，附加仿真专用字段：
      sim_server_url     仿真服务器地址
      sim_timeout        HTTP 请求超时 (s)
      sim_retries        HTTP 失败重试次数
      global_camera      全局摄像头配置
      arm_camera_objects arm_id → 该臂工作区内模拟物块列表
    """
    system_config: SystemConfig
    sim_server_url: str = "http://192.168.1.121:8080"
    sim_timeout: float = 15.0
    sim_retries: int = 2
    global_camera: GlobalCameraConfig = field(default_factory=GlobalCameraConfig)
    arm_camera_objects: Dict[int, List[Dict]] = field(default_factory=dict)
    task_rules: Dict[int, Dict[str, dict]] = field(default_factory=dict)
    """arm_id → {object_type → 任务规则 dict (label/position_xy/release_xyz ...)}"""

    # ── 便捷访问 ──
    @property
    def arms(self) -> List[ArmConfig]:
        return self.system_config.arms

    def get_arm(self, arm_id: int) -> Optional[ArmConfig]:
        return self.system_config.get_arm(arm_id)

    def get_camera_objects(self, arm_id: int) -> List[Dict]:
        """返回某臂的模拟物块配置，缺失时回退到全局默认物块。"""
        return self.arm_camera_objects.get(arm_id, [])


def load_sim_config(json_path: str) -> SimSystemConfig:
    """从 config_sim_3arms.json 加载三臂仿真配置。

    参数:
        json_path: config_sim_3arms.json 路径

    返回:
        SimSystemConfig 实例
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # 构造真机 SystemConfig (arm 段直接复用 load_config 的解析逻辑)
    system_config = load_config(json_path)

    # 全局摄像头配置
    gc_raw = raw.get('global_camera', {})
    global_camera = GlobalCameraConfig(
        width=gc_raw.get('width', 640),
        height=gc_raw.get('height', 480),
        table_depth_m=gc_raw.get('table_depth_m', 0.50),
        noise_std_m=gc_raw.get('noise_std_m', 0.0015),
        jitter_px=gc_raw.get('jitter_px', 3.0),
        objects=gc_raw.get('objects', []),
    )

    # 每臂模拟物块 (sim_camera_objects 字段)
    arm_camera_objects = {}
    # 每臂任务规则 (task_rules: object_type → 释放位置等)
    task_rules = {}
    for a in raw.get('arms', []):
        aid = a.get('arm_id')
        objs = a.get('sim_camera_objects', [])
        if aid is not None:
            arm_camera_objects[aid] = objs
            task_rules[aid] = a.get('task_rules', {})

    sim_cfg = SimSystemConfig(
        system_config=system_config,
        sim_server_url=raw.get('sim_server_url', 'http://192.168.1.121:8080'),
        sim_timeout=raw.get('sim_timeout', 15.0),
        sim_retries=raw.get('sim_retries', 2),
        global_camera=global_camera,
        arm_camera_objects=arm_camera_objects,
        task_rules=task_rules,
    )

    logger.info("已加载 %d 个臂的仿真配置 (server=%s)",
                len(system_config.arms), sim_cfg.sim_server_url)
    return sim_cfg


# ═══════════════════════════════════════════════════════════════════════════════
# SimCoordinator
# ═══════════════════════════════════════════════════════════════════════════════

class SimCoordinator:
    """三臂仿真协调器。

    内部包装 MultiArmCoordinator (真机协调器)，通过 __getattr__ 委托全部
    区域锁 / 安全监控 / 状态同步 / 死锁检测方法，因此对 SimArmController
    与 SimVisualizer 而言，SimCoordinator 可当作 MultiArmCoordinator 使用。

    仿真专用能力:
      - 共享 SimulationClient (所有臂共用同一 HTTP 会话)
      - 任务发送串行化: 同一时刻仅一个臂在 POST /task
      - 全局抓取冷却: 相邻两次抓取任务间隔 >= global_cooldown_ms
      - 生产任务落盘: send_task 后由 TaskRecorder 记录每次任务
    """

    def __init__(self, sim_cfg: SimSystemConfig,
                 client: Optional[SimulationClient] = None,
                 task_recorder: Optional[TaskRecorder] = None):
        """
        参数:
            sim_cfg:        三臂仿真配置
            client:         共享 SimulationClient，为 None 时自动创建
            task_recorder:  生产任务落盘记录器，为 None 时不落盘
        """
        self._sim_cfg = sim_cfg
        self._sys_cfg = sim_cfg.system_config

        # 创建共享仿真客户端
        if client is None:
            client = SimulationClient(
                base_url=sim_cfg.sim_server_url,
                timeout=sim_cfg.sim_timeout,
                retries=sim_cfg.sim_retries,
            )
        self._client = client

        # 生产任务落盘记录器
        self._task_recorder = task_recorder

        # 内部真机协调器
        self._coord = MultiArmCoordinator(self._sys_cfg)

        # 任务发送串行化
        self._send_lock = threading.Lock()
        self._last_global_grasp_time = 0.0

    # ── 属性访问 ──

    @property
    def client(self) -> SimulationClient:
        return self._client

    @property
    def config(self) -> SystemConfig:
        return self._sys_cfg

    @property
    def sim_config(self) -> SimSystemConfig:
        return self._sim_cfg

    # ── 委托给内部 MultiArmCoordinator ──

    def __getattr__(self, name):
        """将未定义的属性/方法调用委托给内部 MultiArmCoordinator。"""
        return getattr(self._coord, name)

    def register_arm(self, arm_id: int) -> threading.Event:
        """注册一个臂，返回该臂的停止信号 Event。"""
        return self._coord.register_arm(arm_id)

    def broadcast_stop(self, source_arm_id: Optional[int] = None):
        """广播全局停止。"""
        self._coord.broadcast_stop(source_arm_id)

    # ── 仿真专用: 任务发送 ──

    def send_task(self, arm_id: int, sequence: list,
                  wait_cooldown: bool = True,
                  meta: Optional[dict] = None) -> bool:
        """串行化 + 全局冷却地发送抓取任务序列到仿真服务器，并落盘记录。

        参数:
            arm_id:         发送任务的臂编号
            sequence:       任务序列 (由 TaskBuilder 构建)
            wait_cooldown:  是否等待全局冷却 (默认 True)
            meta:           可选任务上下文 dict (如 object_type / goal /
                            release_xyz)，会合并进落盘记录；未配置 TaskRecorder
                            时该参数被忽略

        返回:
            True  发送成功
            False 发送失败 (网络错误 / 服务器 5xx)
        """
        with self._send_lock:
            # 全局抓取冷却: 所有臂共用，相邻任务间隔 >= global_cooldown_ms
            if wait_cooldown:
                cd = self._sys_cfg.global_cooldown_ms / 1000.0
                elapsed = time.monotonic() - self._last_global_grasp_time
                if elapsed < cd:
                    wait = cd - elapsed
                    logger.debug("Arm-%d: 等待全局冷却 %.2fs", arm_id, wait)
                    time.sleep(wait)

            success = False
            try:
                resp = self._client.post_task(sequence, arm_id=arm_id)
                self._last_global_grasp_time = time.monotonic()
                logger.info("Arm-%d: 任务已发送 (%d 步) — 响应: %s",
                            arm_id, len(sequence), resp)
                success = True
            except SimulationClientError as e:
                logger.error("Arm-%d: 发送任务失败: %s", arm_id, e)
                success = False

            # 生产任务落盘 (成功/失败都记录，便于事后排查)
            if self._task_recorder is not None:
                arm_cfg = self._sys_cfg.get_arm(arm_id)
                record = {
                    "arm_id": arm_id,
                    "arm_name": arm_cfg.name if arm_cfg else f"Arm-{arm_id}",
                    "success": success,
                    "steps": len(sequence),
                }
                if meta:
                    record.update(meta)
                record["sequence"] = sequence
                self._task_recorder.record(record)

            return success

    def check_connection(self) -> bool:
        """检查仿真服务器连通性。"""
        return self._client.check_connection()

    def close(self):
        """关闭共享仿真客户端。"""
        try:
            self._client.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python sim_coordinator.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("SimCoordinator 自测")
    print("=" * 60)

    # 1. 配置加载
    print("\n[1] 配置加载...")
    cfg_path = os.path.join(_simulation_dir, 'config_sim_3arms.json')
    sim_cfg = load_sim_config(cfg_path)
    print(f"    server: {sim_cfg.sim_server_url}")
    print(f"    arms:   {len(sim_cfg.arms)}")
    for arm in sim_cfg.arms:
        objs = sim_cfg.get_camera_objects(arm.arm_id)
        print(f"    Arm-{arm.arm_id} ({arm.name}): zone={arm.workspace_zone} "
              f"detect={arm.detect_xyz} 物块={len(objs)}")
    print(f"    全局摄像头: {sim_cfg.global_camera.width}x{sim_cfg.global_camera.height} "
          f"物块={len(sim_cfg.global_camera.objects)}")
    assert len(sim_cfg.arms) == 3
    print("    [PASS]")

    # 2. SimCoordinator 实例化 (无需连接服务器)
    print("\n[2] SimCoordinator 实例化...")
    coord = SimCoordinator(sim_cfg)
    evt = coord.register_arm(0)
    print(f"    Arm-0 已注册, stop_event={type(evt).__name__}")
    assert coord.get_arm_state(0).value == 'IDLE'
    print("    [PASS]")

    # 3. __getattr__ 委托验证
    print("\n[3] 方法委托验证...")
    summary = coord.get_summary()
    assert 'arm_states' in summary and 'zones' in summary
    print(f"    get_summary() -> keys: {list(summary.keys())}")
    print("    [PASS]")

    coord.close()
    print("\n自测完成 [PASS]")
