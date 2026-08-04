#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_3arm_grasp — 三臂协同抓取启动器
====================================
主入口脚本：加载配置 → 启动 Coordinator → 启动 3 个 ArmController →
启动可视化 → 等待退出 → 清理。

用法:
    python run_3arm_grasp.py [配置文件路径]

    默认配置文件: config_3arms.json (同目录)
    快捷键:
      q / ESC  退出程序
      r        清除 HAZARD 区域（需人工确认安全后按下）
      s        打印系统状态摘要

硬件要求:
    - 3x Intel RealSense D435 (每臂一个，USB3.0 独立通道)
    - 3x UFACTORY xArm Lite 6 (同一局域网)
    - 1x 控制电脑 (Windows / Linux)
"""

import os
import sys
import time
import logging
import signal

# 确保可以导入 multi_arm 包
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.join(_current_dir, '..')
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from multi_arm.config import load_config
from multi_arm.coordinator import MultiArmCoordinator
from multi_arm.arm_controller import ArmController
from multi_arm.visualizer import MultiArmVisualizer


def setup_logging():
    """配置日志：同时输出到控制台和文件。"""
    log_dir = os.path.join(_current_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # 文件输出（含 DEBUG 级别详细信息）
    log_file = os.path.join(
        log_dir,
        f"3arm_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    return log_file


def main():
    # ═══════════════════════════════════════════════════════════════════
    # 1. 解析命令行参数
    # ═══════════════════════════════════════════════════════════════════
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(_current_dir, 'config_3arms.json')

    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        print(f"用法: python {sys.argv[0]} [config_path]")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # 2. 初始化日志
    # ═══════════════════════════════════════════════════════════════════
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("三臂协同抓取系统 启动中...")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════════════
    # 3. 加载配置
    # ═══════════════════════════════════════════════════════════════════
    try:
        config = load_config(config_path)
        logger.info(f"已加载 {len(config.arms)} 个臂的配置:")
        for arm in config.arms:
            logger.info(
                f"  Arm-{arm.arm_id} ({arm.name}): {arm.robot_ip} "
                f"观察位={arm.detect_xyz} 释放位={arm.release_xyz}"
            )
    except Exception as e:
        logger.exception("配置加载失败")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # 4. 启动 Coordinator
    # ═══════════════════════════════════════════════════════════════════
    logger.info("初始化 Coordinator...")
    coordinator = MultiArmCoordinator(config)

    # ═══════════════════════════════════════════════════════════════════
    # 5. 创建 ArmController 实例
    # ═══════════════════════════════════════════════════════════════════
    controllers = []
    for arm_cfg in config.arms:
        logger.info(
            f"创建 ArmController-{arm_cfg.arm_id} ({arm_cfg.name})..."
        )
        ctrl = ArmController(arm_cfg, config, coordinator)
        controllers.append(ctrl)

    # ═══════════════════════════════════════════════════════════════════
    # 6. 初始化 Visualizer
    # ═══════════════════════════════════════════════════════════════════
    logger.info("初始化 Visualizer...")
    viz = MultiArmVisualizer(config, coordinator)
    for ctrl in controllers:
        viz.attach_controller(ctrl)

    # ═══════════════════════════════════════════════════════════════════
    # 7. 启动所有线程
    # ═══════════════════════════════════════════════════════════════════
    logger.info("启动所有子系统...")
    for ctrl in controllers:
        ctrl.start()
        time.sleep(1.0)  # 错开初始化，减少 USB/网络瞬时负载

    viz.start()
    logger.info("=" * 60)
    logger.info("系统运行中 — 按 q 或 ESC 退出")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════════════
    # 8. 等待退出信号
    # ═══════════════════════════════════════════════════════════════════
    def _signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在退出...")
        coordinator.broadcast_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # 主线程阻塞等待可视化退出（用户按 q 或 ESC）
        viz.join()
    except KeyboardInterrupt:
        logger.info("收到 KeyboardInterrupt")
    finally:
        # ═══════════════════════════════════════════════════════════════
        # 9. 清理
        # ═══════════════════════════════════════════════════════════════
        logger.info("正在关闭系统...")
        coordinator.broadcast_stop()

        # 停止所有臂线程
        for ctrl in controllers:
            ctrl.stop()
        for ctrl in controllers:
            ctrl.join(timeout=10.0)

        viz.stop()
        viz.join(timeout=5.0)

        # 打印最终统计
        counts = coordinator.get_grasp_counts()
        total = sum(counts.values())
        logger.info("=" * 60)
        logger.info(f"系统关闭。总抓取次数: {total}")
        for arm_id, cnt in sorted(counts.items()):
            arm_name = config.get_arm(arm_id)
            name_str = arm_name.name if arm_name else f"Arm-{arm_id}"
            logger.info(f"  {name_str}: {cnt} 次")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
