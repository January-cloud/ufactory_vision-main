#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sim_3arm — 三臂协同仿真启动器
====================================
仿真平台上的三臂协同抓取主入口。结构对齐真机 multi_arm/run_3arm_grasp.py，
将硬件调用替换为仿真调用，并额外加入全局俯瞰摄像头。

用法:
    python run_sim_3arm.py
    python run_sim_3arm.py --server http://192.168.1.121:8080
    python run_sim_3arm.py --sim-camera          # 从仿真服务器获取每臂图像
    python run_sim_3arm.py --no-camera           # 无相机测试模式 (免 GGCNN/服务器)
    python run_sim_3arm.py --config config_sim_3arms.json
    python run_sim_3arm.py --task-log my_tasks.jsonl   # 自定义任务落盘路径
    python run_sim_3arm.py --no-task-log               # 禁用生产任务落盘
    python run_sim_3arm.py --ext-input                 # 外部坐标输入模式
    python run_sim_3arm.py --ext-input --ext-port 9090 # 自定义外部输入端口

流程:
    加载配置 → 创建任务落盘记录器 → 连接仿真服务器 → 创建全局摄像头 →
    创建 SimCoordinator → 创建 3× SimArmController → 创建 SimVisualizer →
    启动全部线程 → 等待退出 → 清理并打印统计 (含落盘任务数)。

按键:
    q / ESC  退出程序
    r        清除 HAZARD 区域
    s        打印系统状态摘要
"""

import os
import sys
import time
import logging
import argparse
import signal

# ── 包内路径引导 ──
_sim_3arm_dir = os.path.dirname(os.path.abspath(__file__))
_simulation_dir = os.path.dirname(_sim_3arm_dir)
_demo_dir = os.path.dirname(_simulation_dir)
for _p in (_sim_3arm_dir, _simulation_dir, _demo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulation.simulation_client import SimulationClient
from simulation.global_camera import GlobalCamera
from simulation.external_input import ExternalInputServer

from simulation.sim_3arm.sim_coordinator import SimCoordinator, load_sim_config
from simulation.sim_3arm.sim_arm_controller import SimArmController
from simulation.sim_3arm.sim_visualizer import SimVisualizer
from simulation.sim_3arm.task_recorder import TaskRecorder


def setup_logging(log_dir: str):
    """配置日志：同时输出到控制台和文件。"""
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    log_file = os.path.join(
        log_dir,
        f"sim3arm_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    return log_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='三臂协同仿真 — 视觉引导抓取 (仿真平台)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python run_sim_3arm.py
  python run_sim_3arm.py --server http://192.168.1.121:8080
  python run_sim_3arm.py --sim-camera
  python run_sim_3arm.py --no-camera
  python run_sim_3arm.py --task-log logs/tasks.jsonl
  python run_sim_3arm.py --no-task-log
  python run_sim_3arm.py --ext-input              外部坐标输入模式
  python run_sim_3arm.py --ext-input --ext-port 9090
        """
    )
    parser.add_argument('--config', default=None,
                        help='三臂仿真配置文件 (默认: config_sim_3arms.json)')
    parser.add_argument('--server', default=None,
                        help='仿真服务器 URL (覆盖配置文件)')
    parser.add_argument('--timeout', type=float, default=None,
                        help='HTTP 超时秒数 (覆盖配置文件)')
    parser.add_argument('--model', default=None,
                        help='GGCNN2 模型权重路径覆盖')
    parser.add_argument('--sim-camera', action='store_true',
                        help='从仿真服务器获取每臂摄像头图像 (POST /get_camera)')
    parser.add_argument('--no-camera', action='store_true',
                        help='无相机模式 — 虚拟深度图，仅测试线程/通信链路')
    parser.add_argument('--task-log', default=None,
                        help='生产任务落盘路径 (.jsonl)，默认 <sim_3arm>/logs/'
                             'tasks_<时间>.jsonl')
    parser.add_argument('--no-task-log', action='store_true',
                        help='禁用生产任务落盘')
    parser.add_argument('--ext-input', action='store_true',
                        help='外部坐标输入模式 — 所有臂共用一台 HTTP 服务器，'
                             'POST /grasp_target 携带 arm_id 路由 (跳过 GGCNN)')
    parser.add_argument('--ext-port', type=int, default=8090,
                        help='外部坐标输入 HTTP 服务器端口 (默认: 8090)')
    parser.add_argument('--ext-host', type=str, default='0.0.0.0',
                        help='外部坐标输入 HTTP 服务器绑定地址 (默认: 0.0.0.0)')
    return parser.parse_args()


def main():
    args = parse_args()

    # ═══════════════════════════════════════════════════════════════════
    # 1. 加载配置
    # ═══════════════════════════════════════════════════════════════════
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(_simulation_dir, 'config_sim_3arms.json')

    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    log_file = setup_logging(os.path.join(_sim_3arm_dir, 'logs'))
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("三臂协同仿真系统 启动中...")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)

    sim_cfg = load_sim_config(config_path)
    if args.server:
        sim_cfg.sim_server_url = args.server
    if args.timeout:
        sim_cfg.sim_timeout = args.timeout

    # ═══════════════════════════════════════════════════════════════════
    # 2. 创建生产任务落盘记录器
    # ═══════════════════════════════════════════════════════════════════
    task_recorder = None
    if args.no_task_log:
        logger.info("已禁用生产任务落盘 (--no-task-log)")
    else:
        task_log_path = args.task_log
        if not task_log_path:
            task_log_path = os.path.join(
                _sim_3arm_dir, 'logs',
                f"tasks_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
            )
        task_recorder = TaskRecorder(task_log_path)
        logger.info(f"生产任务将落盘到: {task_recorder.path}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. 创建仿真客户端 + 检查连通性
    # ═══════════════════════════════════════════════════════════════════
    client = SimulationClient(
        base_url=sim_cfg.sim_server_url,
        timeout=sim_cfg.sim_timeout,
        retries=sim_cfg.sim_retries,
    )
    logger.info("检查仿真服务器连通性 %s ...", sim_cfg.sim_server_url)
    if not client.check_connection():
        logger.warning(
            "无法连接仿真服务器 %s — 程序将继续运行，"
            "抓取任务将发送失败（可改用 --server 指定地址）。",
            sim_cfg.sim_server_url
        )

    # ═══════════════════════════════════════════════════════════════════
    # 4. 创建全局摄像头
    # ═══════════════════════════════════════════════════════════════════
    gc = sim_cfg.global_camera
    global_cam = GlobalCamera(
        width=gc.width, height=gc.height,
        table_depth_m=gc.table_depth_m,
        objects=gc.objects,
    )
    logger.info(f"全局摄像头初始化完成: {gc.width}x{gc.height} "
                f"物块={len(gc.objects)}")

    # ═══════════════════════════════════════════════════════════════════
    # 4.5 外部坐标输入服务器 (可选)
    # ═══════════════════════════════════════════════════════════════════
    input_server = None
    if args.ext_input:
        logger.info(f"外部坐标输入模式: 启动 HTTP 服务器 "
                    f"{args.ext_host}:{args.ext_port} ...")
        input_server = ExternalInputServer(
            host=args.ext_host, port=args.ext_port
        )
        try:
            input_server.start()
            logger.info(
                "外部坐标输入服务已启动: %s "
                "(POST /grasp_target 携带 arm_id 路由, GET /status)",
                input_server.url
            )
        except OSError as e:
            logger.error(
                "无法启动外部坐标服务器: %s (端口 %d 可能被占用)",
                e, args.ext_port
            )
            client.close()
            sys.exit(1)
    else:
        logger.info("未启用外部坐标输入 (如需使用加 --ext-input)")

    # ═══════════════════════════════════════════════════════════════════
    # 5. 创建 SimCoordinator
    # ═══════════════════════════════════════════════════════════════════
    logger.info("初始化 SimCoordinator...")
    coordinator = SimCoordinator(sim_cfg, client=client,
                                 task_recorder=task_recorder,
                                 ext_input_server=input_server)

    # ═══════════════════════════════════════════════════════════════════
    # 6. 创建 3× SimArmController
    # ═══════════════════════════════════════════════════════════════════
    controllers = []
    for arm_cfg in sim_cfg.arms:
        logger.info(f"创建 SimArmController-{arm_cfg.arm_id} "
                    f"({arm_cfg.name})...")
        ctrl = SimArmController(
            sim_cfg, arm_cfg, coordinator,
            use_sim_camera=args.sim_camera,
            no_camera=args.no_camera,
            model_file=args.model,
            use_ext_input=args.ext_input,
        )
        controllers.append(ctrl)

    # ═══════════════════════════════════════════════════════════════════
    # 7. 初始化 SimVisualizer
    # ═══════════════════════════════════════════════════════════════════
    logger.info("初始化 SimVisualizer...")
    viz = SimVisualizer(sim_cfg.system_config, coordinator, global_cam)
    for ctrl in controllers:
        viz.attach_controller(ctrl)

    # ═══════════════════════════════════════════════════════════════════
    # 8. 启动所有线程
    # ═══════════════════════════════════════════════════════════════════
    logger.info("启动所有子系统...")
    for ctrl in controllers:
        ctrl.start()
        time.sleep(0.5)  # 错开初始化，减少 CPU 瞬时负载

    viz.start()
    logger.info("=" * 60)
    logger.info("系统运行中 — 按 q 或 ESC 退出")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════════════
    # 9. 等待退出信号
    # ═══════════════════════════════════════════════════════════════════
    def _signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在退出...")
        coordinator.broadcast_stop()

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, AttributeError):
        pass

    try:
        viz.join()
    except KeyboardInterrupt:
        logger.info("收到 KeyboardInterrupt")
    finally:
        # ═══════════════════════════════════════════════════════════════
        # 10. 清理
        # ═══════════════════════════════════════════════════════════════
        logger.info("正在关闭系统...")
        coordinator.broadcast_stop()

        for ctrl in controllers:
            ctrl.stop()
        for ctrl in controllers:
            ctrl.join(timeout=5.0)

        viz.stop()
        viz.join(timeout=5.0)

        coordinator.close()

        # 关闭任务落盘记录器
        if task_recorder is not None:
            task_recorder.close()
            logger.info(
                f"生产任务已落盘: {task_recorder.count} 条 "
                f"→ {task_recorder.path}"
            )

        # 关闭外部坐标输入服务器
        if input_server is not None:
            input_server.stop()
            logger.info("外部坐标输入服务器已关闭")

        # 打印最终统计
        counts = coordinator.get_grasp_counts()
        total = sum(counts.values())
        logger.info("=" * 60)
        logger.info(f"系统关闭。总抓取次数: {total}")
        for arm_id, cnt in sorted(counts.items()):
            arm_cfg = sim_cfg.get_arm(arm_id)
            name_str = arm_cfg.name if arm_cfg else f"Arm-{arm_id}"
            logger.info(f"  {name_str}: {cnt} 次")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
