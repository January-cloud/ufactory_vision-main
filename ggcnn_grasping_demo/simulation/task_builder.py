#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_builder — 抓取任务序列构建器
==================================
将 GGCNN 推理得到的基坐标系抓取目标转换为仿真平台可执行的动作序列。
每个序列包含一系列 move（移动）和 vacuum（真空吸盘）动作，
附带 wait 字段用于模拟动作完成后的等待时间。

设计依据:
    run_rs_d435_grasp_lite6_new_best.py 的 two_stage_grasp() 阶段3
    (搬运 → 放置 → 回观察位) 的完整动作序列。
    仿真模式下相机固定，跳过阶段2（移到目标正上方重拍精定位）。
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 配置数据类
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimGraspConfig:
    """仿真抓取任务配置 — 与现有单臂脚本的全局常量对应。

    所有距离单位为 mm，时间单位为秒，角度单位为度。
    """
    # ── 关键位姿 ──
    detect_xyz: List[float] = field(default_factory=lambda: [200.0, 0.0, 380.0])
    """观察位 [x, y, z]，抓取完成后回到此位置"""

    release_xyz: List[float] = field(default_factory=lambda: [225.0, -89.0, 83.0])
    """释放/放置点 [x, y, z]"""

    # ── 抓取参数 ──
    above_z: float = 300.0
    """安全悬停高度 mm — 在目标上方接近但不接触"""

    grasping_min_z: float = 70.0
    """最低允许下降 Z (防撞桌) mm — 实际抓取 Z 不低于此值"""

    # ── 等待时间 ──
    move_wait: float = 0.5
    """每次 move 完成后额外等待秒数（模拟静置防抖）"""

    descend_wait: float = 1.0
    """下降动作完成后额外等待秒数（低速下降需要更多时间）"""

    suction_wait: float = 0.8
    """开启真空后等待负压建立的秒数"""

    release_wait: float = 0.5
    """关闭真空后等待释放的秒数"""


# ═══════════════════════════════════════════════════════════════════════════════
# TaskBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TaskBuilder:
    """将基坐标系抓取目标转换为仿真动作序列。

    使用方式:
        config = SimGraspConfig()
        builder = TaskBuilder(config)

        # 构建完整 pick-and-place 序列
        seq = builder.build_pick_and_place([230, -50, 85, 180, 0, -30])

        # 或手动构建单步动作
        move = builder.build_move(200, 0, 380)
        vac  = builder.build_vacuum(on=True)
    """

    def __init__(self, config: Optional[SimGraspConfig] = None):
        """
        参数:
            config: 抓取配置，为 None 时使用默认值
        """
        self.config = config if config is not None else SimGraspConfig()

    # ── 公开方法 ──────────────────────────────────────────────────────────────

    def build_pick_and_place(self, grasp_target: List[float],
                             release_xyz: Optional[List[float]] = None) -> list:
        """根据抓取目标构建完整的 pick-and-place 任务序列。

        参数:
            grasp_target: [X, Y, Z, Roll, Pitch, Yaw] — 基坐标系抓取目标
                          X, Y, Z 单位 mm，Roll, Pitch, Yaw 单位度
                          典型值: [230, -50, 85, 180, 0, -30]
            release_xyz:  可选，覆盖默认释放位置 [x, y, z] mm。
                          用于按物体类型切换释放位置（不同任务放到不同料位）。
                          None 时使用 self.config.release_xyz。

        返回:
            任务 dict 列表，每项包含 type, params, wait 字段

        生成的序列 (10 步):
            1. move 到目标正上方 (above_z) — 对准 XY + 抓取角度
            2. move 维持上方静置 (stay)
            3. move 直线下降到抓取高度 (Z + 安全限位)
            4. vacuum ON — 吸取物体
            5. move 抬升至观察高度 (保持抓取角度)
            6. move 平移到释放点正上方 (yaw 归零)
            7. move 下降到释放高度
            8. vacuum OFF — 释放物体
            9. move 抬升至观察高度
           10. move 回到观察位
        """
        cfg = self.config
        X, Y, Z = grasp_target[0], grasp_target[1], grasp_target[2]
        Yaw = grasp_target[5] if len(grasp_target) > 5 else 0.0

        # 抓取 Z 安全限位
        grasp_z = max(Z, cfg.grasping_min_z)

        detect_z = cfg.detect_xyz[2]
        if release_xyz is not None:
            release_x, release_y, release_z = release_xyz
        else:
            release_x, release_y, release_z = cfg.release_xyz

        sequence = []

        # Step 1: 从观察位移到目标正上方（安全高度，对准抓取角度）
        sequence.append(self.build_move(
            X, Y, cfg.above_z,
            roll=180, pitch=0, yaw=Yaw,
            wait=cfg.move_wait
        ))

        # Step 2: 上方静置（消抖） — 重复同一个位置
        sequence.append(self.build_move(
            X, Y, cfg.above_z,
            roll=180, pitch=0, yaw=Yaw,
            wait=cfg.move_wait
        ))

        # Step 3: 直线下降至抓取高度
        sequence.append(self.build_move(
            X, Y, grasp_z,
            roll=180, pitch=0, yaw=Yaw,
            wait=cfg.descend_wait
        ))

        # Step 4: 开启真空吸盘
        sequence.append(self.build_vacuum(on=True, wait=cfg.suction_wait))

        # Step 5: 抬升至观察高度（保持抓取角度）
        sequence.append(self.build_move(
            X, Y, detect_z,
            roll=180, pitch=0, yaw=Yaw,
            wait=cfg.move_wait
        ))

        # Step 6: 平移到释放点正上方
        sequence.append(self.build_move(
            release_x, release_y, detect_z,
            roll=180, pitch=0, yaw=0,
            wait=cfg.move_wait
        ))

        # Step 7: 下降到释放高度
        sequence.append(self.build_move(
            release_x, release_y, release_z,
            roll=180, pitch=0, yaw=0,
            wait=cfg.descend_wait
        ))

        # Step 8: 关闭真空吸盘
        sequence.append(self.build_vacuum(on=False, wait=cfg.release_wait))

        # Step 9: 抬升至观察高度
        sequence.append(self.build_move(
            release_x, release_y, detect_z,
            roll=180, pitch=0, yaw=0,
            wait=cfg.move_wait
        ))

        # Step 10: 回到观察位
        sequence.append(self.build_move(
            cfg.detect_xyz[0], cfg.detect_xyz[1], detect_z,
            roll=180, pitch=0, yaw=0,
            wait=cfg.move_wait
        ))

        return sequence

    def build_goto_observe(self) -> list:
        """构建回到观察位的序列。

        返回:
            包含单个 move 动作的列表
        """
        return [self.build_move(
            self.config.detect_xyz[0],
            self.config.detect_xyz[1],
            self.config.detect_xyz[2],
            roll=180, pitch=0, yaw=0,
            wait=self.config.move_wait,
        )]

    def build_move(self, x: float, y: float, z: float,
                   roll: float = 180.0, pitch: float = 0.0, yaw: float = 0.0,
                   wait: Optional[float] = None) -> dict:
        """构建单个 move 动作。

        参数:
            x, y, z:        目标位置 (mm)
            roll, pitch, yaw: 目标姿态 (度)
            wait:            动作完成后等待秒数，None 则使用默认 move_wait

        返回:
            {"type": "move", "params": {...}, "wait": ...}
        """
        if wait is None:
            wait = self.config.move_wait

        return {
            "type": "move",
            "params": {
                "x": round(x, 1),
                "y": round(y, 1),
                "z": round(z, 1),
                "roll": round(roll, 1),
                "pitch": round(pitch, 1),
                "yaw": round(yaw, 1),
            },
            "wait": round(wait, 2),
        }

    def build_vacuum(self, on: bool, wait: Optional[float] = None) -> dict:
        """构建单个 vacuum 动作。

        参数:
            on:   True=开启真空, False=关闭真空
            wait: 动作完成后等待秒数，None 则使用默认值

        返回:
            {"type": "vacuum", "params": {"on": bool}, "wait": ...}
        """
        if wait is None:
            wait = self.config.suction_wait if on else self.config.release_wait

        return {
            "type": "vacuum",
            "params": {"on": on},
            "wait": round(wait, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python task_builder.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("TaskBuilder 自测")
    print("=" * 60)

    # 使用默认配置
    builder = TaskBuilder()

    # 模拟一个抓取目标: [X=230, Y=-50, Z=85, Roll=180, Pitch=0, Yaw=-30]
    grasp_target = [230.0, -50.0, 85.0, 180.0, 0.0, -30.0]

    print(f"\n[1] 抓取目标: {grasp_target}")
    print(f"    配置: detect={builder.config.detect_xyz}, release={builder.config.release_xyz}")
    print(f"          above_z={builder.config.above_z}, grasping_min_z={builder.config.grasping_min_z}")

    # 构建完整 pick-and-place 序列
    sequence = builder.build_pick_and_place(grasp_target)
    print(f"\n[2] 生成的 pick-and-place 序列 ({len(sequence)} 步):")

    envelope = {"sequence": sequence}
    json_str = json.dumps(envelope, indent=2, ensure_ascii=False)
    print(json_str)

    # 验证关键点
    print(f"\n[3] 验证:")
    steps_ok = True
    # Step 1: above_z 高度
    assert sequence[0]["params"]["z"] == 300.0, "Step1 Z should be above_z"
    # Step 3: grasp Z 应 >= grasping_min_z
    assert sequence[2]["params"]["z"] == max(85.0, 70.0), "Step3 Z should be max(target_z, min_z)"
    # Step 4: vacuum ON
    assert sequence[3]["type"] == "vacuum" and sequence[3]["params"]["on"] is True, "Step4 should be vacuum ON"
    # Step 8: vacuum OFF
    assert sequence[7]["type"] == "vacuum" and sequence[7]["params"]["on"] is False, "Step8 should be vacuum OFF"
    # Step 10: 回到观察位
    assert sequence[9]["params"]["x"] == 200.0 and sequence[9]["params"]["y"] == 0.0, "Step10 should return to observe"

    print("    [PASS] 所有断言通过")

    # 测试 build_goto_observe
    print(f"\n[4] build_goto_observe():")
    goto = builder.build_goto_observe()
    print(f"    {json.dumps(goto, indent=4)}")

    # 测试安全限位（目标 Z 低于 grasping_min_z 时）
    print(f"\n[5] 安全限位测试 (target Z=30 < min_z=70):")
    low_target = [200.0, 0.0, 30.0, 180.0, 0.0, 0.0]
    low_seq = builder.build_pick_and_place(low_target)
    grasp_step = low_seq[2]
    print(f"    原始 Z=30 → 限位后 Z={grasp_step['params']['z']}")
    assert grasp_step["params"]["z"] == 70.0, "Z should be clamped to min"
    print("    [PASS] 安全限位生效")

    print(f"\n自测完成 [PASS]")
