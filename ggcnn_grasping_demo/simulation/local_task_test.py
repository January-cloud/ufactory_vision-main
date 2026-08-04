#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地任务序列模拟执行器
======================
不需要仿真服务器，从本地 JSON 文件读取 /task 格式的任务序列，
逐步模拟执行并打印每个动作的详细信息。

用法:
    # 使用内置示例任务
    python local_task_test.py

    # 从 JSON 文件读取
    python local_task_test.py --input my_task.json

    # 先生成示例 JSON 文件
    python local_task_test.py --export sample_task.json

JSON 文件格式 (与 /task 接口一致):
{
    "sequence": [
        {"type": "move", "params": {"x": 230, "y": -50, "z": 300, "roll": 180, "pitch": 0, "yaw": -30}, "wait": 0.5},
        {"type": "vacuum", "params": {"on": true}, "wait": 0.8}
    ]
}
"""

import json
import time
import math
import argparse
import os
import sys

# ── 颜色输出 (Windows 终端兼容) ──
try:
    import colorama
    colorama.init()
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
except ImportError:
    GREEN = YELLOW = CYAN = MAGENTA = RED = RESET = BOLD = ''


# ═══════════════════════════════════════════════════════════════════════════════
# 内置示例任务序列
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_TASK = {
    "sequence": [
        {"type": "move", "params": {"x": 200.0, "y": 0.0, "z": 380.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}, "wait": 0.5},
        {"type": "move", "params": {"x": 230.0, "y": -50.0, "z": 300.0, "roll": 180.0, "pitch": 0.0, "yaw": -30.0}, "wait": 0.5},
        {"type": "move", "params": {"x": 230.0, "y": -50.0, "z": 300.0, "roll": 180.0, "pitch": 0.0, "yaw": -30.0}, "wait": 0.5},
        {"type": "move", "params": {"x": 230.0, "y": -50.0, "z": 85.0, "roll": 180.0, "pitch": 0.0, "yaw": -30.0}, "wait": 1.0},
        {"type": "vacuum", "params": {"on": True}, "wait": 0.8},
        {"type": "move", "params": {"x": 230.0, "y": -50.0, "z": 380.0, "roll": 180.0, "pitch": 0.0, "yaw": -30.0}, "wait": 0.5},
        {"type": "move", "params": {"x": 225.0, "y": -89.0, "z": 380.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}, "wait": 0.5},
        {"type": "move", "params": {"x": 225.0, "y": -89.0, "z": 83.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}, "wait": 1.0},
        {"type": "vacuum", "params": {"on": False}, "wait": 0.5},
        {"type": "move", "params": {"x": 225.0, "y": -89.0, "z": 380.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}, "wait": 0.5},
        {"type": "move", "params": {"x": 200.0, "y": 0.0, "z": 380.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}, "wait": 0.5},
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# 轨迹追踪
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryTracker:
    """追踪末端执行器的位置变化，用于统计和检查。"""

    def __init__(self, start_pos=(200, 0, 380)):
        self.positions = [start_pos]  # (x, y, z)
        self.total_distance = 0.0

    def record(self, x, y, z):
        prev = self.positions[-1]
        dist = math.sqrt((x - prev[0])**2 + (y - prev[1])**2 + (z - prev[2])**2)
        self.total_distance += dist
        self.positions.append((x, y, z))
        return dist

    def stats(self):
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        zs = [p[2] for p in self.positions]
        return {
            "waypoints": len(self.positions),
            "total_distance_mm": self.total_distance,
            "x_range": (min(xs), max(xs)),
            "y_range": (min(ys), max(ys)),
            "z_range": (min(zs), max(zs)),
            "min_z": min(zs),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 模拟执行
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_execution(sequence, speed_factor=1.0, verbose=True):
    """逐步模拟执行任务序列。

    参数:
        sequence: 任务列表
        speed_factor: 加速因子 (>1 加快, 0 表示跳过等待)
        verbose: 是否打印详细信息
    """
    tracker = TrajectoryTracker()
    total_wait = 0.0
    move_count = 0
    vacuum_on_count = 0
    vacuum_off_count = 0

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  本地模拟执行开始{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"  任务步数: {len(sequence)}")
    print(f"  加速因子: {speed_factor}x")
    print(f"  起始位置: ({tracker.positions[0][0]:.0f}, {tracker.positions[0][1]:.0f}, {tracker.positions[0][2]:.0f}) mm\n")

    for i, step in enumerate(sequence):
        step_type = step.get("type", "unknown")
        params = step.get("params", {})
        wait = step.get("wait", 0.0)
        total_wait += wait

        # ── 绘制步骤分隔线 ──
        print(f"  {CYAN}── 步骤 {i+1}/{len(sequence)}{RESET} ", end="")

        if step_type == "move":
            move_count += 1
            x, y, z = params.get("x", 0), params.get("y", 0), params.get("z", 0)
            roll, pitch, yaw = params.get("roll", 180), params.get("pitch", 0), params.get("yaw", 0)
            dist = tracker.record(x, y, z)

            print(f"{GREEN}MOVE{RESET}  -> ({x:7.1f}, {y:7.1f}, {z:7.1f}) mm  "
                  f"rpy=({roll:.0f}, {pitch:.0f}, {yaw:.0f}) deg")

            if verbose:
                prev = tracker.positions[-2]
                dz = z - prev[2]
                arrow = "v" if dz < -5 else ("^" if dz > 5 else ">")
                print(f"         d=({x-prev[0]:+.1f}, {y-prev[1]:+.1f}, {z-prev[2]:+.1f}) mm  "
                      f"dist={dist:.1f} mm  {arrow}  wait={wait}s")

                # 检查安全限位
                if z < 70:
                    print(f"         {RED}[WARN] Z={z:.0f} < 安全限位 70mm!{RESET}")

        elif step_type == "vacuum":
            on = params.get("on", False)
            if on:
                vacuum_on_count += 1
                print(f"{YELLOW}VACUUM ON{RESET}   {'-'*40}  wait={wait}s")
                if verbose:
                    print(f"         吸取物体，等待负压建立 ({wait}s)")
            else:
                vacuum_off_count += 1
                print(f"{MAGENTA}VACUUM OFF{RESET}  {'-'*40}  wait={wait}s")
                if verbose:
                    print(f"         释放物体 ({wait}s)")

        else:
            print(f"{RED}UNKNOWN: {step_type}{RESET}")

        # ── 模拟等待 ──
        if speed_factor > 0 and wait > 0:
            actual_wait = wait / speed_factor
            time.sleep(actual_wait)

    # ── 汇总 ──
    stats = tracker.stats()
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  执行完成 — 统计汇总{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"  move 动作:     {move_count} 次")
    print(f"  vacuum ON:     {vacuum_on_count} 次")
    print(f"  vacuum OFF:    {vacuum_off_count} 次")
    actual_time = total_wait / speed_factor if speed_factor > 0 else 0
    print(f"  总等待时间:    {total_wait:.1f}s (模拟实际耗时 {actual_time:.1f}s)")
    print(f"  路径点:        {stats['waypoints']} 个")
    print(f"  总行程:        {stats['total_distance_mm']:.1f} mm")
    print(f"  X 范围:        [{stats['x_range'][0]:.0f}, {stats['x_range'][1]:.0f}] mm")
    print(f"  Y 范围:        [{stats['y_range'][0]:.0f}, {stats['y_range'][1]:.0f}] mm")
    print(f"  Z 范围:        [{stats['z_range'][0]:.0f}, {stats['z_range'][1]:.0f}] mm")
    print(f"  最低 Z:        {stats['min_z']:.0f} mm", end="")
    if stats['min_z'] < 70:
        print(f"  {RED}[WARN] 低于安全限位!{RESET}")
    else:
        print(f"  {GREEN}[OK] 安全{RESET}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# JSON 验证
# ═══════════════════════════════════════════════════════════════════════════════

def validate_sequence(sequence):
    """验证任务序列格式是否正确。"""
    errors = []

    if not isinstance(sequence, list):
        return ["sequence 必须是列表"]

    valid_types = {"move", "vacuum"}

    for i, step in enumerate(sequence):
        if not isinstance(step, dict):
            errors.append(f"步骤 {i+1}: 不是 dict")
            continue

        step_type = step.get("type")
        if step_type not in valid_types:
            errors.append(f"步骤 {i+1}: 未知类型 '{step_type}'")
            continue

        params = step.get("params", {})

        if step_type == "move":
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
                if key not in params:
                    errors.append(f"步骤 {i+1} (move): 缺少参数 '{key}'")
            if params.get("z", 100) < 0:
                errors.append(f"步骤 {i+1} (move): Z={params['z']} 为负值，可能撞穿桌面")

        elif step_type == "vacuum":
            if "on" not in params:
                errors.append(f"步骤 {i+1} (vacuum): 缺少参数 'on'")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='本地任务序列模拟执行器 — 不需要仿真服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python local_task_test.py                        # 使用内置示例
  python local_task_test.py --input my_task.json   # 从文件读取
  python local_task_test.py --export sample.json   # 导出示例 JSON
  python local_task_test.py --speed 5.0            # 5倍速执行
  python local_task_test.py --speed 0              # 跳过等待瞬间完成
        """
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入 JSON 文件路径（不指定则使用内置示例）')
    parser.add_argument('--export', '-e', type=str, default=None,
                        help='导出示例任务 JSON 到文件')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='模拟速度因子 (默认 1.0, 0=跳过等待)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='简洁模式（不打印每步详情）')
    args = parser.parse_args()

    # ── 导出模式 ──
    if args.export:
        export_path = args.export
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_TASK, f, indent=2, ensure_ascii=False)
        print(f"{GREEN}[OK] 示例任务已导出到: {export_path}{RESET}")
        print(f"  包含 {len(SAMPLE_TASK['sequence'])} 个步骤")
        return

    # ── 加载任务 ──
    if args.input:
        print(f"从文件加载: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        print(f"{CYAN}使用内置示例任务（完整 pick-and-place 10步序列）{RESET}")
        data = SAMPLE_TASK

    # ── 提取序列（支持 {"sequence": [...]} 和裸列表两种格式） ──
    if isinstance(data, dict) and "sequence" in data:
        sequence = data["sequence"]
    elif isinstance(data, list):
        sequence = data
    else:
        print(f"{RED}错误: JSON 格式不正确，需要 {{\"sequence\": [...]}} 或裸列表{RESET}")
        sys.exit(1)

    # ── 验证 ──
    errors = validate_sequence(sequence)
    if errors:
        print(f"\n{RED}[WARN] JSON 格式验证发现 {len(errors)} 个问题:{RESET}")
        for e in errors:
            print(f"  - {e}")
        if any("缺少参数" in e for e in errors):
            print(f"\n{RED}存在严重错误，终止执行。请检查 JSON 文件格式。{RESET}")
            sys.exit(1)
    else:
        print(f"{GREEN}[OK] JSON 格式验证通过{RESET}")

    # ── 打印 JSON 预览 ──
    print(f"\n{BOLD}任务 JSON 预览:{RESET}")
    print(json.dumps(data if isinstance(data, dict) else {"sequence": sequence},
                     indent=2, ensure_ascii=False))
    print()

    # ── 模拟执行 ──
    simulate_execution(sequence, speed_factor=args.speed, verbose=not args.quiet)


if __name__ == '__main__':
    main()
