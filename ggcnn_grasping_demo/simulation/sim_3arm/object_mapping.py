#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
object_mapping — 物体识别与任务切换映射
=========================================
将"识别出的物体类型"映射为"抓取任务参数"（当前主要是释放位置）。

设计意图:
    GGCNN 只输出几何抓取点，无法区分物体类别。本模块提供一个可替换的
    识别接口 (ObjectMapper.recognize)：当前阶段用坐标最近邻匹配来模拟
    识别（任务 A 对应 (0,0) 物块、任务 B 对应 (20,0)、任务 C 对应 (40,0)），
    后续可直接把 recognize() 内部实现替换为视觉分类器，调用方无需改动。

数据流:
    grasp target (base 系 XY, mm) → ObjectMapper.recognize()
        → TaskRule(含 release_xyz) → TaskBuilder.build_pick_and_place(
              grasp_target, release_xyz=rule.release_xyz)
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── 包内路径引导 ──
_sim_3arm_dir = os.path.dirname(os.path.abspath(__file__))
_simulation_dir = os.path.dirname(_sim_3arm_dir)
_demo_dir = os.path.dirname(_simulation_dir)
for _p in (_sim_3arm_dir, _simulation_dir, _demo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════════
# 任务规则
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskRule:
    """一条任务规则：某种物体 → 对应的抓取任务参数。

    属性:
        object_type:     物体类型标识 (e.g. "part_a")
        label:           显示名 (e.g. "Part A")
        position_xy:     物体底座中心在臂 base 坐标系下的 (X, Y)，单位 mm
        match_radius_mm: 坐标匹配半径，超出即视为不匹配，单位 mm
        release_xyz:     该物体对应的释放位置 [x, y, z] mm；None 时用臂默认值
        # 未来可扩展: approach_speed / descend_speed / above_z / grasp_retry ...
    """
    object_type: str
    label: str = ""
    position_xy: Tuple[float, float] = (0.0, 0.0)
    match_radius_mm: float = 30.0
    release_xyz: Optional[List[float]] = None


def task_rule_from_dict(d: dict) -> "TaskRule":
    """从配置字典构造 TaskRule。"""
    return TaskRule(
        object_type=d['object_type'],
        label=d.get('label', d['object_type']),
        position_xy=tuple(d.get('position_xy', [0.0, 0.0])),
        match_radius_mm=float(d.get('match_radius_mm', 30.0)),
        release_xyz=d.get('release_xyz'),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 物体识别 + 任务切换映射
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectMapper:
    """物体识别 + 任务切换的接口封装。

    用法:
        mapper = ObjectMapper(rules)
        rule = mapper.recognize(grasp_x, grasp_y)      # 相机模式：坐标匹配
        rule = mapper.recognize_by_type("part_a")      # 手动模式：按类型查表

    未来接入视觉识别时，只需替换 recognize() 内部的匹配逻辑，
    调用方（SimArmController）无需改动。
    """

    def __init__(self, rules: Optional[List[TaskRule]] = None):
        self._rules: List[TaskRule] = rules or []

    @property
    def rules(self) -> List[TaskRule]:
        return self._rules

    def add_rule(self, rule: TaskRule):
        """添加一条任务规则。"""
        self._rules.append(rule)

    # ── 公开接口 ──

    def recognize(self, grasp_x: float, grasp_y: float) -> Optional[TaskRule]:
        """根据抓取目标 (base 系 XY, mm) 识别最近的物体类型。

        未来视觉实现示例:
            object_type = self._vision_classifier.classify(color_img, depth_img)
            return self.recognize_by_type(object_type)

        当前实现:
            返回 match_radius 内距离最近的 TaskRule；无匹配返回 None。
        """
        return self._match_by_position(grasp_x, grasp_y)

    def recognize_by_type(self, object_type: str) -> Optional[TaskRule]:
        """按物体类型直接查表（用于外部输入/manual 模式指定 label）。"""
        for r in self._rules:
            if r.object_type == object_type:
                return r
        return None

    def get_release_xyz(self, object_type: Optional[str],
                        default_release: List[float]) -> List[float]:
        """根据物体类型返回释放位置；无匹配或未配置时回退默认值。"""
        if object_type:
            rule = self.recognize_by_type(object_type)
            if rule and rule.release_xyz:
                return list(rule.release_xyz)
        return list(default_release)

    # ── 内部实现 ──

    def _match_by_position(self, grasp_x: float,
                           grasp_y: float) -> Optional[TaskRule]:
        """最近邻坐标匹配：在 match_radius 内找到距离最近的规则。"""
        best = None
        best_dist = float('inf')
        for r in self._rules:
            dx = grasp_x - r.position_xy[0]
            dy = grasp_y - r.position_xy[1]
            dist = math.hypot(dx, dy)
            if dist <= r.match_radius_mm and dist < best_dist:
                best = r
                best_dist = dist
        return best


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python object_mapping.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("ObjectMapper 自测")
    print("=" * 60)

    rules = [
        TaskRule(object_type="part_a", label="Part A",
                 position_xy=(0.0, 0.0), release_xyz=[150, -400, 83]),
        TaskRule(object_type="part_b", label="Part B",
                 position_xy=(20.0, 0.0), release_xyz=[150, -250, 83]),
        TaskRule(object_type="part_c", label="Part C",
                 position_xy=(40.0, 0.0), release_xyz=[150, -100, 83]),
    ]
    mapper = ObjectMapper(rules)

    # 1. 坐标匹配
    print("\n[1] 坐标匹配...")
    r = mapper.recognize(3.0, 2.0)
    assert r is not None and r.object_type == "part_a", f"匹配失败: {r}"
    print(f"    grasp(3,2) -> {r.object_type} ({r.label}) release={r.release_xyz}")
    r = mapper.recognize(22.0, -1.0)
    assert r is not None and r.object_type == "part_b", f"匹配失败: {r}"
    print(f"    grasp(22,-1) -> {r.object_type} ({r.label}) release={r.release_xyz}")
    r = mapper.recognize(43.0, 3.0)
    assert r is not None and r.object_type == "part_c", f"匹配失败: {r}"
    print(f"    grasp(43,3) -> {r.object_type} ({r.label}) release={r.release_xyz}")
    print("    [PASS]")

    # 2. 超出半径 → None
    print("\n[2] 半径过滤...")
    r = mapper.recognize(100.0, 100.0)
    assert r is None, f"应无匹配: {r}"
    print("    grasp(100,100) -> None  [PASS]")

    # 3. 按类型查表
    print("\n[3] 按类型查表...")
    r = mapper.recognize_by_type("part_b")
    assert r is not None and r.release_xyz == [150, -250, 83]
    r = mapper.recognize_by_type("nonexistent")
    assert r is None
    print("    part_b -> release [150, -250, 83]; nonexistent -> None  [PASS]")

    # 4. get_release_xyz 回退默认
    print("\n[4] get_release_xyz 回退默认...")
    default = [225, -89, 83]
    assert mapper.get_release_xyz("part_c", default) == [150, -100, 83]
    assert mapper.get_release_xyz(None, default) == default
    assert mapper.get_release_xyz("unknown", default) == default
    print("    类型命中用规则值，否则回退默认  [PASS]")

    print("\n自测完成 [PASS]")
