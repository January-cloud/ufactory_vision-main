#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_controller — 单臂控制线程
==============================
封装完整的单臂抓取管线（基于 run_rs_d435_grasp_lite6_new_best.py），
增加与中心协调器的双向通信和安全移动包装。

每个 ArmController 实例运行在独立线程中，拥有:
  - 自己的 RealSense D435 相机实例
  - 自己的 TorchGGCNN 模型实例
  - 自己的 XArmAPI 机械臂连接
  - 与 Coordinator 的双向状态同步

复用原始脚本的函数逻辑，将全局变量替换为实例属性。
"""

import os
import sys
import cv2
import time
import math
import threading
import logging
import numpy as np
from queue import Queue
from collections import Counter
from typing import Optional, List, Tuple

# 添加路径以导入现有模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.join(_current_dir, '..')
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from camera.rs_camera import RealSenseCamera
from camera.utils import get_combined_img
from grasp.ggcnn_torch import TorchGGCNN
from grasp.helpers.matrix_funcs import euler2mat, convert_pose
from xarm.wrapper import XArmAPI

from .config import ArmConfig, SystemConfig, ArmState
from .coordinator import MultiArmCoordinator
from .dominant_cluster import dominant_cluster
from .collision_avoidance import point_in_zone, interarm_distance

logger = logging.getLogger(__name__)


class ArmController:
    """
    单臂控制器 — 在独立线程中运行完整的抓取管线。

    使用方式:
        config = load_config("config_3arms.json")
        coord = MultiArmCoordinator(config)
        ctrl = ArmController(config.arms[0], config, coord)
        ctrl.start()   # 启动线程
        ...
        ctrl.stop()    # 通知停止
        ctrl.join()    # 等待线程结束

    线程内部流程（与 run_rs_d435_grasp_lite6_new_best.py 完全一致）:
      1. 初始化硬件（相机 + GGCNN模型 + 机械臂连接）
      2. 移到观察位
      3. 主循环:
         a. 获取 RGB-D 图像 → GGCNN 推理
         b. 候选缓冲 + 聚类 → 判断是否 LOCKED
         c. 稳定锁定 → 请求协调区权限 → 两段式抓取
         d. 更新 Coordinator 状态和位置
      4. 清理资源
    """

    def __init__(self, arm_config: ArmConfig, system_config: SystemConfig,
                 coordinator: MultiArmCoordinator):
        self._cfg = arm_config
        self._sys_cfg = system_config
        self._coord = coordinator

        # ── 硬件句柄（在线程中初始化） ──
        self._arm: Optional[XArmAPI] = None
        self._camera: Optional[RealSenseCamera] = None
        self._ggcnn: Optional[TorchGGCNN] = None
        self._K: Optional[np.ndarray] = None  # 相机内参 3×3 矩阵

        # ── 线程控制 ──
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # ── 候选缓冲（主循环中持续维护） ──
        self._cand_buf: List[List[float]] = []
        self._grasp_count: int = 0
        self._last_grasp_time: float = 0.0

        # ── 可视化帧缓冲区（供 visualizer 线程读取） ──
        self._latest_color: Optional[np.ndarray] = None
        self._latest_heatmap: Optional[np.ndarray] = None
        self._latest_state_label: str = "INIT"
        self._frame_lock = threading.Lock()

    # ═══════════════════════════════════════════════════════════════════
    # 线程生命周期
    # ═══════════════════════════════════════════════════════════════════

    def start(self):
        """启动臂控制线程。"""
        if self._running:
            return
        self._stop_event = self._coord.register_arm(self._cfg.arm_id)
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"ArmCtrl-{self._cfg.arm_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"ArmController-{self._cfg.arm_id} 线程已启动")

    def stop(self):
        """请求停止臂线程。"""
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        logger.info(f"ArmController-{self._cfg.arm_id} 停止请求已发出")

    def join(self, timeout: Optional[float] = None):
        """等待线程结束。"""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def grasp_count(self) -> int:
        """当前成功抓取次数。"""
        return self._grasp_count

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray],
                                         Optional[np.ndarray], str, int]:
        """获取最新帧数据（供 visualizer 定时拉取）。"""
        with self._frame_lock:
            return (self._latest_color, self._latest_heatmap,
                    self._latest_state_label, self._grasp_count)

    # ═══════════════════════════════════════════════════════════════════
    # 主运行入口
    # ═══════════════════════════════════════════════════════════════════

    def _run(self):
        """线程主函数。初始化硬件 → 移动到观察位 → 主循环 → 清理。"""
        try:
            self._init_hardware()
            self._goto_observation()
            self._main_loop()
        except Exception:
            logger.exception(f"ArmController-{self._cfg.arm_id} 致命错误")
            self._coord.update_arm_state(self._cfg.arm_id, ArmState.STOPPED)
        finally:
            self._cleanup()

    # ═══════════════════════════════════════════════════════════════════
    # 硬件初始化
    # ═══════════════════════════════════════════════════════════════════

    def _init_hardware(self):
        """初始化相机、GGCNN 模型、机械臂连接。"""
        arm_id = self._cfg.arm_id

        # ① 初始化 D435 相机
        logger.info(f"Arm-{arm_id}: 初始化 D435 相机...")
        self._camera = RealSenseCamera(
            width=self._cfg.cam_width,
            height=self._cfg.cam_height,
        )
        ci, _ = self._camera.get_intrinsics()
        self._K = np.array([
            [ci.fx, 0, ci.ppx],
            [0, ci.fy, ci.ppy],
            [0, 0, 1],
        ])

        # ② 加载 GGCNN2 模型
        model_path = os.path.normpath(os.path.join(
            _current_dir, '..', 'models', 'epoch_50_cornell'
        )) if not self._cfg.model_file else self._cfg.model_file
        logger.info(f"Arm-{arm_id}: 加载 GGCNN 模型 {model_path}...")
        self._ggcnn = TorchGGCNN({
            'MODEL_FILE': model_path,
            'OPEN_LOOP_HEIGHT': 0,          # 始终取全局最大值（开环模式）
            'GGCNN_IN_THREAD': False,       # 主线程同步推理
            'DEPTH_CAM_K': self._K,
        }, Queue(1), Queue(1))
        time.sleep(1)

        # ③ 连接机械臂
        logger.info(f"Arm-{arm_id}: 连接机械臂 {self._cfg.robot_ip}...")
        self._arm = XArmAPI(self._cfg.robot_ip, report_type='real')
        self._arm.motion_enable(True)
        self._arm.clean_error()
        self._arm.set_mode(0)               # 位置控制模式
        self._arm.set_state(0)              # 就绪状态
        time.sleep(0.5)
        self._arm.set_vacuum_gripper(on=False)
        logger.info(f"Arm-{arm_id}: 硬件初始化完成")

    # ═══════════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════════

    def _main_loop(self):
        """
        主事件循环 — 与原脚本 main() 中的 while 循环逻辑一致。
        每帧执行：图像获取 → GGCNN推理 → 候选聚类 → 触发抓取。
        """
        cfg = self._cfg
        arm_id = cfg.arm_id

        # 预览裁剪参数
        cs = min(cfg.cam_height, cfg.cam_width)
        off_r = max(0, cfg.cam_height - cs) // 2
        off_c = max(0, cfg.cam_width - cs) // 2

        logger.info(f"Arm-{arm_id}: 进入主循环 (观察位: {cfg.detect_xyz})")

        while (self._running and self._arm is not None
               and self._arm.connected
               and not self._stop_event.is_set()):

            # ── 错误检测与恢复 ──
            if self._arm.error_code != 0:
                self._handle_fault(arm_id)
                self._cand_buf.clear()
                self._last_grasp_time = time.monotonic()
                continue

            # ── 发布当前位置到 Coordinator ──
            eef = self._get_eef_pose_m()
            eef_mm = [eef[0]*1000, eef[1]*1000, eef[2]*1000,
                       eef[3], eef[4], eef[5]]
            self._coord.update_arm_position(arm_id, tuple(eef_mm))

            # ── 获取 RGB-D 图像 + GGCNN 推理 ──
            color_image, depth_image = self._camera.get_images(align=True)
            depth_image = depth_image.astype(np.float32)
            grasp_img, result = self._ggcnn.get_grasp_img(
                depth_image, self._K, eef[2]
            )

            color_crop = color_image[off_r:off_r+cs, off_c:off_c+cs].copy()

            # ── 候选缓冲 + 聚类分析 ──
            goal = None
            stable = False

            if result is not None and result[2] > cfg.min_result_z_mm / 1000.0:
                # 将相机系结果转到基坐标系
                cand = self._cam_result_to_base(eef, result)
                self._cand_buf.append(cand)

                # 保持滑动窗口大小
                if len(self._cand_buf) > cfg.cand_window:
                    self._cand_buf.pop(0)

                # 缓冲区满后执行聚类
                if len(self._cand_buf) >= cfg.cand_window:
                    pick, cnt = dominant_cluster(
                        self._cand_buf, cfg.cand_bin_mm
                    )
                    if cnt >= cfg.cand_min_frames:
                        stable = True
                        goal = pick
                    else:
                        goal = cand
                else:
                    goal = cand

                # 在彩色图上绘制抓取点标记
                mp = self._ggcnn.prev_mp
                cv2.circle(
                    color_crop, (int(mp[1]), int(mp[0])), 6,
                    (0, 255, 0) if stable else (0, 200, 255), -1
                )
            else:
                # 无有效检测 → 清空缓冲
                self._cand_buf.clear()

            # ── 更新 Coordinator 状态 ──
            if stable:
                self._coord.update_arm_state(arm_id, ArmState.LOCKED)
            else:
                self._coord.update_arm_state(arm_id, ArmState.IDLE)

            # ── 触发抓取 ──
            if (stable and self._in_own_zone(goal)
                    and (time.monotonic() - self._last_grasp_time) > cfg.cooldown_sec):

                # 判断是否需要协调区权限
                need_coord_zone = self._which_coord_zone(goal[:2])

                if need_coord_zone and not self._coord.request_zone(
                    arm_id, need_coord_zone
                ):
                    # 协调区被占用，尝试将目标微移到独占区内
                    alt_goal = self._snap_to_exclusive_zone(goal)
                    if alt_goal is not None:
                        goal = alt_goal
                        logger.info(
                            f"Arm-{arm_id}: 协调区忙，"
                            f"已微移到独占区 X={goal[0]:.1f} Y={goal[1]:.1f}"
                        )
                    else:
                        logger.debug(
                            f"Arm-{arm_id}: 协调区忙，放弃本轮"
                        )
                        self._publish_frame(color_crop, grasp_img, "WAIT_ZONE")
                        continue

                # 执行两段式抓取
                self._coord.update_arm_state(arm_id, ArmState.GRASPING)
                success = self._two_stage_grasp_wrapped(
                    self._camera, self._ggcnn, self._K, self._arm, goal
                )

                if success:
                    self._grasp_count += 1
                    self._coord.record_grasp(arm_id)

                # 释放协调区
                if need_coord_zone:
                    self._coord.release_zone(arm_id, need_coord_zone)

                self._last_grasp_time = time.monotonic()
                self._cand_buf.clear()
                self._coord.update_arm_state(arm_id, ArmState.IDLE)

            # ── 状态栏渲染 ──
            state_label = ("LOCKED" if stable
                           else ("SEARCHING" if goal is not None
                                 else "NO OBJECT"))
            cv2.putText(
                color_crop,
                f"{cfg.name} | {state_label} | grasped:{self._grasp_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

            self._publish_frame(color_crop, grasp_img, state_label)

        # 循环结束
        logger.info(f"Arm-{arm_id}: 主循环退出")

    # ═══════════════════════════════════════════════════════════════════
    # 坐标变换（复用原脚本逻辑）
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_eef_pose_m(arm: XArmAPI) -> List[float]:
        """读取机械臂当前末端位姿，返回 [x, y, z, roll, pitch, yaw] (m, rad)。"""
        _, p = arm.get_position(is_radian=True)
        return [p[0]*0.001, p[1]*0.001, p[2]*0.001, p[3], p[4], p[5]]

    def _cam_result_to_base(self, eef_pose: List[float],
                            result: List[float]) -> List[float]:
        """
        将 GGCNN 相机系抓取点转换到机械臂基坐标系。

        变换链: Base ← EEF(当前位置) ← ColorCamera(手眼标定) ← DepthCamera
        返回 [x, y, z, 180, 0, yaw] (mm, deg)。
        """
        cfg = self._cfg
        x, y, z, ang = result[0], result[1], result[2], result[3]
        gp = [x, y, z, 0, 0, -1 * ang]

        # 连乘变换矩阵
        mat = (euler2mat(eef_pose)
               * euler2mat(cfg.euler_eef_to_color_opt)
               * euler2mat(cfg.euler_color_to_depth_opt))
        gp_base = convert_pose(gp, mat)

        # 抓取角度归一化到 [-π, 0]
        if gp_base[5] < -np.pi:
            gp_base[5] += np.pi
        elif gp_base[5] > 0:
            gp_base[5] -= np.pi

        return [
            gp_base[0]*1000, gp_base[1]*1000,
            gp_base[2]*1000 + cfg.gripper_z_mm,
            180, 0, math.degrees(gp_base[5] + np.pi/2),
        ]

    def _camera_offset_base_xy(self, roll: float, pitch: float,
                                yaw: float) -> Tuple[float, float]:
        """
        计算相机相对法兰在基坐标系下的 XY 偏移 (mm)。

        用于阶段2：已知物块在基系下的粗坐标，反算法兰应移到哪里
        才能使相机恰好位于物块正上方。
        """
        R = np.array(euler2mat([0, 0, 0, roll, pitch, yaw]))[:3, :3]
        cam_off_eef = np.array(
            self._cfg.euler_eef_to_color_opt[:3]
        ).reshape(3, 1)
        off = np.array(R @ cam_off_eef).flatten() * 1000.0
        return off[0], off[1]

    # ═══════════════════════════════════════════════════════════════════
    # 安全移动包装
    # ═══════════════════════════════════════════════════════════════════

    def _safe_move(self, x: float, y: float, z: float,
                   roll: float = 180, pitch: float = 0, yaw: float = 0,
                   speed: Optional[int] = None, wait: bool = True,
                   state_during: ArmState = ArmState.MOVING) -> bool:
        """
        安全的机械臂移动。

        1. 先通过 Coordinator 检查目标位置安全性
        2. 发布移动状态
        3. 执行移动命令
        4. 移动完成后更新位置到 Coordinator

        返回 True 表示移动成功，False 表示被安全检查阻止。
        """
        arm_id = self._cfg.arm_id
        if speed is None:
            speed = self._cfg.move_speed

        # 安全检查
        if not self._coord.is_safe_to_move(arm_id, (x, y, z)):
            logger.warning(
                f"Arm-{arm_id}: 安全检查阻止移动到 ({x:.0f},{y:.0f},{z:.0f})"
            )
            return False

        # 发布状态
        self._coord.update_arm_state(arm_id, state_during)

        try:
            self._arm.set_position(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                speed=speed, wait=wait,
            )
            # 移动完成后刷新位置
            if wait:
                eef = self._get_eef_pose_m()
                self._coord.update_arm_position(
                    arm_id,
                    (eef[0]*1000, eef[1]*1000, eef[2]*1000,
                     eef[3], eef[4], eef[5])
                )
            return True
        except Exception:
            logger.exception(f"Arm-{arm_id}: 移动失败")
            return False

    def _safe_move_z(self, z: float, speed: Optional[int] = None,
                     wait: bool = True) -> bool:
        """仅 Z 轴移动（保持当前 XY 和姿态不变）。"""
        if speed is None:
            speed = self._cfg.descend_speed
        _, pos = self._arm.get_position()
        return self._safe_move(
            pos[0], pos[1], z,
            roll=pos[3], pitch=pos[4], yaw=pos[5],
            speed=speed, wait=wait,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 区域检查
    # ═══════════════════════════════════════════════════════════════════

    def _in_own_zone(self, goal: List[float]) -> bool:
        """判断抓取目标是否在有效工作区内（独占区或可申请的协调区）。"""
        x, y = goal[0], goal[1]
        cfg = self._cfg

        # 在独占区内
        if point_in_zone(x, y, cfg.workspace_zone):
            return True

        # 在协调区内
        for cz in cfg.coordination_zones:
            if point_in_zone(x, y, cz):
                return True

        return False

    def _which_coord_zone(self, xy: Tuple[float, float]) -> Optional[str]:
        """判断 (x, y) 在哪个协调区内，返回区域名称。不在协调区返回 None。"""
        for i, cz in enumerate(self._cfg.coordination_zones):
            if point_in_zone(xy[0], xy[1], cz):
                return f"coord_{self._cfg.arm_id}_{i}"
        return None

    def _snap_to_exclusive_zone(
            self, goal: List[float]) -> Optional[List[float]]:
        """
        当目标在协调区但无法获取权限时，尝试将抓取点微调到独占区内。

        如果目标靠近协调区与独占区的 Y 边界（在 margin 范围内），
        向独占区方向偏移一小段距离，从而避开协调区。
        """
        cfg = self._cfg
        x, y = goal[0], goal[1]
        zone = cfg.workspace_zone
        margin = 30.0  # 微调范围 (mm)

        # 目标在协调区但靠近独占区 Y 下边界
        if y < zone[2] and y > zone[2] - margin:
            snapped = list(goal)
            snapped[1] = zone[2] + 5  # 放入独占区
            return snapped
        # 目标在协调区但靠近独占区 Y 上边界
        if y > zone[3] and y < zone[3] + margin:
            snapped = list(goal)
            snapped[1] = zone[3] - 5
            return snapped

        return None

    # ═══════════════════════════════════════════════════════════════════
    # 故障处理
    # ═══════════════════════════════════════════════════════════════════

    def _handle_fault(self, arm_id: int):
        """
        分级故障处理。特别处理协调区内的故障：

        第一步：尝试抬升到安全高度（优先抬升，避免影响其他臂）
        第二步：释放所有协调区锁
        第三步：执行标准错误恢复
        第四步：回到观察位
        """
        logger.warning(f"Arm-{arm_id}: 检测到错误 code={self._arm.error_code}")

        self._coord.update_arm_state(arm_id, ArmState.RECOVERING)

        # 第一步：尝试安全抬升
        current_z = self._arm.get_position()[1][2]
        safe_z = self._cfg.detect_xyz[2]
        try:
            if current_z < safe_z:
                self._arm.set_position(
                    z=safe_z, speed=self._cfg.move_speed, wait=True
                )
        except Exception:
            logger.exception(f"Arm-{arm_id}: 紧急抬升失败")

        # 第二步：释放所有协调区
        self._coord.release_all_zones(arm_id)

        # 第三步：标准恢复
        self._recover()

        # 第四步：回到观察位
        try:
            self._arm.set_position(z=safe_z, speed=self._cfg.move_speed, wait=True)
            self._arm.set_position(
                x=self._cfg.detect_xyz[0],
                y=self._cfg.detect_xyz[1],
                z=self._cfg.detect_xyz[2],
                roll=180, pitch=0, yaw=0,
                speed=self._cfg.move_speed, wait=True,
            )
        except Exception:
            logger.exception(f"Arm-{arm_id}: 回观察位失败")

        self._coord.update_arm_state(arm_id, ArmState.IDLE)
        logger.info(f"Arm-{arm_id}: 故障恢复完成")

    def _recover(self):
        """标准错误恢复：清警 → 清错 → 重新使能 → 关吸盘。"""
        arm = self._arm
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        arm.set_vacuum_gripper(on=False)

    # ═══════════════════════════════════════════════════════════════════
    # 两段式抓取（封装原脚本 two_stage_grasp）
    # ═══════════════════════════════════════════════════════════════════

    def _two_stage_grasp_wrapped(self, camera, ggcnn, K, arm, g1) -> bool:
        """
        两段式抓取：阶段2（精定位）+ 阶段3（下降+吸取+搬运+放置）。

        与原脚本 two_stage_grasp() 逻辑一致，增加:
          - 每步移动前安全检查
          - 协调区锁定期续约
          - 状态实时发布到 Coordinator
        """
        cfg = self._cfg
        arm_id = cfg.arm_id

        # 计算拍照姿态下相机相对法兰的偏移
        ox, oy = self._camera_offset_base_xy(math.pi, 0, 0)
        aim = list(g1)

        # 续约定时器
        last_renew = time.monotonic()

        for attempt in range(cfg.grasp_retry + 1):
            # 定期续约持有的协调区锁
            self._renew_zones_if_needed(last_renew)
            last_renew = time.monotonic()

            # ═════════════════════════════════════════════════════════
            # 阶段2：相机移到 aim 正上方 → 停留静置 → 多帧精定位
            # ═════════════════════════════════════════════════════════
            above_target = (aim[0] - ox, aim[1] - oy, cfg.above_z)
            if not self._safe_move(
                above_target[0], above_target[1], above_target[2],
                roll=180, pitch=0, yaw=0,
                speed=cfg.move_speed, state_during=ArmState.MOVING,
            ):
                continue

            # 静置防抖
            time.sleep(cfg.stay_sec_after_move)
            g2 = self._detect_target_robust(camera, ggcnn, K, arm)

            if g2 is None or not self._in_own_zone(g2):
                logger.info(
                    f"Arm-{arm_id}: 阶段2 精定位失败 (第{attempt+1}次)"
                )
                self._safe_move_z(cfg.above_z)
                continue

            # Z 限位保护
            g2[2] = max(g2[2], cfg.grasping_min_z)
            aim = list(g2)
            logger.info(
                f"Arm-{arm_id}: 精定位 X={g2[0]:.1f} Y={g2[1]:.1f} "
                f"Z={g2[2]:.1f} (第{attempt+1}次)"
            )

            # ═════════════════════════════════════════════════════════
            # 阶段3：对准 g2 → 直线下降 → 开真空 → 负压检测
            # ═════════════════════════════════════════════════════════
            self._coord.update_arm_state(arm_id, ArmState.GRASPING)

            # 高位对准目标 XY，姿态朝向抓取角度
            if not self._safe_move(
                g2[0], g2[1], cfg.above_z,
                roll=180, pitch=0, yaw=g2[5],
                speed=cfg.move_speed,
            ):
                continue

            # 直线下降（低速）
            self._coord.update_arm_state(arm_id, ArmState.GRASPING)
            if not self._safe_move(
                g2[0], g2[1], g2[2],
                roll=180, pitch=0, yaw=g2[5],
                speed=cfg.descend_speed,
            ):
                self._safe_move_z(cfg.above_z)
                continue

            # 开真空吸盘
            arm.set_vacuum_gripper(on=True)

            # 负压检测：轮询 TI0 数字输入
            if self._check_suction_ok(arm):
                logger.info(
                    f"Arm-{arm_id}: 吸取成功 (第{attempt+1}次尝试)"
                )

                # ── 搬运 → 放置 → 回观察位 ──
                self._coord.update_arm_state(arm_id, ArmState.PLACING)

                # 抬升至安全高度
                self._safe_move_z(cfg.detect_xyz[2], speed=cfg.move_speed)
                # 移到释放点上方
                self._safe_move(
                    cfg.release_xyz[0], cfg.release_xyz[1],
                    cfg.detect_xyz[2],
                    roll=180, pitch=0, yaw=0,
                    speed=cfg.move_speed,
                )
                # 下降到释放高度
                self._safe_move_z(
                    cfg.release_xyz[2], speed=cfg.descend_speed
                )
                # 关真空放料
                arm.set_vacuum_gripper(on=False)
                time.sleep(0.5)
                # 抬升
                self._safe_move_z(cfg.detect_xyz[2], speed=cfg.move_speed)
                # 回到观察位
                self._goto_observation()
                logger.info(f"Arm-{arm_id}: 抓取完成 [{self._grasp_count+1}]")
                return True

            # ── 空抓处理：关真空 → 抬升 → 回到阶段2重试 ──
            logger.info(
                f"Arm-{arm_id}: 空抓 (第{attempt+1}次)，"
                f"回精确观察位重新识别"
            )
            arm.set_vacuum_gripper(on=False)
            self._safe_move_z(cfg.above_z, speed=cfg.move_speed)

        # 所有尝试失败 → 放弃本轮，回观察位
        logger.warning(f"Arm-{arm_id}: {cfg.grasp_retry+1}次尝试均失败，放弃")
        arm.set_vacuum_gripper(on=False)
        self._safe_move_z(cfg.detect_xyz[2], speed=cfg.move_speed)
        self._goto_observation()
        return False

    # ═══════════════════════════════════════════════════════════════════
    # 精定位（复用原脚本 detect_target_robust）
    # ═══════════════════════════════════════════════════════════════════

    def _detect_target_robust(self, camera, ggcnn, K, arm) -> Optional[List[float]]:
        """
        多帧聚类精定位。与原脚本 detect_target_robust() 一致。

        采集 FINE_FRAMES 帧 → 按 BIN_MM 网格聚类 → 找出现频率最高的格子
        → 用该簇内所有点取均值 → 得到稳健的抓取目标。
        """
        cfg = self._cfg
        goals = []
        cs = min(cfg.cam_height, cfg.cam_width)
        off_r = max(0, cfg.cam_height - cs) // 2
        off_c = max(0, cfg.cam_width - cs) // 2

        for _ in range(cfg.fine_frames):
            color, depth = camera.get_images(align=True)
            depth = depth.astype(np.float32)
            eef = self._get_eef_pose_m(arm)
            gimg, res = ggcnn.get_grasp_img(depth, K, eef[2])
            if res is not None and res[2] > cfg.min_result_z_mm / 1000.0:
                goals.append(self._cam_result_to_base(eef, res))
            cc = color[off_r:off_r+cs, off_c:off_c+cs].copy()
            self._publish_frame(
                cc, cv2.resize(gimg, (cs, cs)), "FINE_LOC"
            )
            time.sleep(cfg.fine_frame_interval)

        # 有效帧数不足 → 失败
        if len(goals) < max(3, cfg.fine_frames // 3):
            return None

        a = np.array(goals)
        # XY 量化到网格
        keys = np.round(a[:, :2] / cfg.bin_mm).astype(int)
        keys_t = [tuple(k) for k in keys]
        best_key, best_cnt = Counter(keys_t).most_common(1)[0]
        # 取主簇及其相邻格
        sel = [i for i, k in enumerate(keys_t)
               if abs(k[0]-best_key[0]) <= 1
               and abs(k[1]-best_key[1]) <= 1]
        cluster = a[sel]
        logger.debug(
            f"Arm-{self._cfg.arm_id}: 精定位有效{len(goals)}帧, "
            f"主簇{len(cluster)}帧"
        )
        g = np.mean(cluster, axis=0)
        return [float(g[0]), float(g[1]), float(g[2]), 180.0, 0.0, float(g[5])]

    # ═══════════════════════════════════════════════════════════════════
    # 吸取检测（复用原脚本 check_suction_ok）
    # ═══════════════════════════════════════════════════════════════════

    def _check_suction_ok(self, arm: XArmAPI) -> bool:
        """
        读末端数字输入 TI0 判断真空是否吸住物料。

        轮询 suction_wait_ms 毫秒，若 TI0 == ti0_ok_value → 吸住成功。
        """
        cfg = self._cfg
        start = time.time()
        while (time.time() - start) < cfg.suction_wait_ms / 1000.0:
            ret = arm.get_tgpio_digital(0)
            code, val = ret[0], ret[1]
            if isinstance(val, (list, tuple)):
                val = val[0]
            if code == 0 and val == cfg.ti0_ok_value:
                return True
            time.sleep(0.05)
        return False

    # ═══════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════

    def _goto_observation(self):
        """移到观察位。"""
        cfg = self._cfg
        self._safe_move(
            cfg.detect_xyz[0], cfg.detect_xyz[1], cfg.detect_xyz[2],
            roll=180, pitch=0, yaw=0,
            speed=cfg.move_speed, state_during=ArmState.MOVING,
        )

    def _renew_zones_if_needed(self, last_renew: float):
        """如果距上次续约已超过间隔，续约所有持有的协调区锁。"""
        if (time.monotonic() - last_renew
                >= self._sys_cfg.lease_renew_interval_s):
            for i in range(len(self._cfg.coordination_zones)):
                zone_name = f"coord_{self._cfg.arm_id}_{i}"
                self._coord.renew_zone(self._cfg.arm_id, zone_name)

    def _publish_frame(self, color: np.ndarray, heatmap: np.ndarray,
                       state_label: str):
        """发布最新帧到缓冲区（供 visualizer 线程读取）。"""
        with self._frame_lock:
            self._latest_color = color.copy() if color is not None else None
            self._latest_heatmap = heatmap.copy() if heatmap is not None else None
            self._latest_state_label = state_label

    def _cleanup(self):
        """清理资源：关吸盘、断机械臂、停相机。"""
        arm_id = self._cfg.arm_id
        logger.info(f"Arm-{arm_id}: 开始清理...")

        self._coord.update_arm_state(arm_id, ArmState.STOPPED)

        if self._arm is not None:
            try:
                self._arm.set_vacuum_gripper(on=False)
                self._arm.disconnect()
            except Exception:
                pass
            self._arm = None

        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

        logger.info(f"Arm-{arm_id}: 清理完成")
