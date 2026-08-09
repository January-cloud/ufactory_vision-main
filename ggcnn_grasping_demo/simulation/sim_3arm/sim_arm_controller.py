#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_arm_controller — 单臂仿真控制器（线程）
=============================================
三臂仿真中每台机械臂的独立控制线程。结构对齐真机 multi_arm/arm_controller.py，
将硬件调用替换为仿真调用：

  真机 ArmController                   仿真 SimArmController
  -------------------------           -------------------------
  XArmAPI (机械臂 SDK)                 SimulationClient.post_task(arm_id, seq)
  RealSenseCamera (D435)              BuiltinCamera / SimCamera
  两段式抓取 (精定位 + 吸料检测)        一段式 (观察位拍照 → 聚类锁定 → 任务序列)
  _safe_move() 安全检查                request_zone() 协调区锁 + 全局冷却

每个 SimArmController 实例运行在独立线程中，拥有自己的相机、GGCNN 模型
和 TaskBuilder，并通过 SimCoordinator 与其余两臂协调区域与抓取时机。
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
from typing import Optional, List, Tuple

# ── 包内路径引导 ──
_sim_3arm_dir = os.path.dirname(os.path.abspath(__file__))
_simulation_dir = os.path.dirname(_sim_3arm_dir)
_demo_dir = os.path.dirname(_simulation_dir)
for _p in (_sim_3arm_dir, _simulation_dir, _demo_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grasp.ggcnn_torch import TorchGGCNN
from grasp.helpers.matrix_funcs import euler2mat, convert_pose

from simulation.task_builder import TaskBuilder, SimGraspConfig
from simulation.builtin_camera import BuiltinCamera
from simulation.sim_camera import SimCamera
from simulation.simulation_client import SimulationClient
from simulation.external_input import ExternalInputServer

from multi_arm.config import ArmConfig, ArmState
from multi_arm.dominant_cluster import dominant_cluster
from multi_arm.collision_avoidance import point_in_zone
from simulation.sim_3arm.sim_coordinator import SimCoordinator, SimSystemConfig
from simulation.sim_3arm.object_mapping import ObjectMapper, task_rule_from_dict


logger = logging.getLogger(__name__)


class SimArmController:
    """单臂仿真控制器 — 在独立线程中运行视觉抓取管线。

    使用方式:
        sim_cfg = load_sim_config("config_sim_3arms.json")
        coord = SimCoordinator(sim_cfg)
        ctrl = SimArmController(sim_cfg, sim_cfg.arms[0], coord)
        ctrl.start()
        ...
        ctrl.stop(); ctrl.join()

    线程流程:
      1. 初始化 (相机 + GGCNN 模型 + TaskBuilder)
      2. 主循环:
         a. 获取 RGB-D → GGCNN 推理
         b. 候选缓冲 + 聚类 → 是否 LOCKED
         c. 稳定锁定 → 协调区权限 → 构建任务序列 → POST /task
         d. 更新 Coordinator 状态 / 发布帧给 SimVisualizer
      3. 清理
    """

    def __init__(self, sim_cfg: SimSystemConfig, arm_cfg: ArmConfig,
                 coordinator: SimCoordinator,
                 use_sim_camera: bool = False,
                 no_camera: bool = False,
                 model_file: Optional[str] = None,
                 use_ext_input: bool = False):
        """
        参数:
            sim_cfg:         三臂仿真配置
            arm_cfg:         该臂配置
            coordinator:     SimCoordinator 实例
            use_sim_camera:  True 使用 SimCamera (HTTP 从仿真服务器取图)，
                             False 使用本地合成 BuiltinCamera
            no_camera:       True 无相机模式 — 使用虚拟深度图，仅测试线程/
                             通信链路，不加载 GGCNN，不产生抓取
            model_file:      GGCNN 权重路径覆盖 (None 用 arm_cfg.model_file)
            use_ext_input:   True 外部坐标输入模式 — 跳过相机/GGCNN，从
                             coordinator 的 ExternalInputServer 轮询外部坐标
                             (POST /grasp_target，按 arm_id 路由)
        """
        self._sim_cfg = sim_cfg
        self._cfg = arm_cfg
        self._coord = coordinator
        self._client: SimulationClient = coordinator.client
        self._use_sim_camera = use_sim_camera
        self._no_camera = no_camera
        self._use_ext_input = use_ext_input

        self._camera = None
        self._ggcnn = None
        self._K: Optional[np.ndarray] = None
        self._builder: Optional[TaskBuilder] = None
        self._object_mapper: ObjectMapper = ObjectMapper()
        self._current_object_type: str = ""

        # 外部坐标输入服务器 (三臂共享，按 arm_id 路由)
        self._ext_server: Optional[ExternalInputServer] = (
            coordinator.ext_input_server if use_ext_input else None
        )

        # 线程控制
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 候选缓冲
        self._cand_buf: List[List[float]] = []
        self._grasp_count = 0
        self._last_grasp_time = 0.0

        # 可视化帧缓冲 (供 SimVisualizer 读取)
        self._latest_color: Optional[np.ndarray] = None
        self._latest_heatmap: Optional[np.ndarray] = None
        self._latest_state_label = "INIT"
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
            name=f"SimArmCtrl-{self._cfg.arm_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"SimArmController-{self._cfg.arm_id} 线程已启动")

    def stop(self):
        """请求停止臂线程。"""
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        logger.info(f"SimArmController-{self._cfg.arm_id} 停止请求已发出")

    def join(self, timeout: Optional[float] = None):
        """等待线程结束。"""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def grasp_count(self) -> int:
        """当前成功抓取次数。"""
        return self._grasp_count

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray],
                                        Optional[np.ndarray], str, str, int]:
        """获取最新帧数据（供 SimVisualizer 定时拉取）。

        返回:
            (color, heatmap, state_label, object_type, grasp_count)
        """
        with self._frame_lock:
            return (self._latest_color, self._latest_heatmap,
                    self._latest_state_label, self._current_object_type,
                    self._grasp_count)

    # ═══════════════════════════════════════════════════════════════════
    # 线程主入口
    # ═══════════════════════════════════════════════════════════════════

    def _run(self):
        """线程主函数。初始化 → 主循环 → 清理。"""
        try:
            self._init()
            self._main_loop()
        except Exception:
            logger.exception(f"SimArmController-{self._cfg.arm_id} 致命错误")
            self._coord.update_arm_state(self._cfg.arm_id, ArmState.STOPPED)
        finally:
            self._cleanup()

    # ═══════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════

    def _init(self):
        """初始化相机、GGCNN 模型、TaskBuilder。

        外部坐标输入模式 (use_ext_input) 下跳过相机与 GGCNN，
        仅构建 TaskBuilder + ObjectMapper，从 ExternalInputServer 轮询坐标。
        """
        arm_id = self._cfg.arm_id

        # ① 外部坐标输入模式：跳过相机与 GGCNN
        if self._use_ext_input:
            if self._ext_server is None:
                raise ValueError(
                    f"Arm-{arm_id}: 启用外部输入模式但 coordinator 未提供 "
                    f"ExternalInputServer — 请检查 run_sim_3arm.py --ext-input"
                )
            logger.info(f"Arm-{arm_id}: 外部坐标输入模式 — 跳过相机/GGCNN")
            self._camera = None
            self._ggcnn = None

            # ③ 构建 TaskBuilder (使用该臂的观察位/释放位)
            self._builder = TaskBuilder(SimGraspConfig(
                detect_xyz=list(self._cfg.detect_xyz),
                release_xyz=list(self._cfg.release_xyz),
                above_z=self._cfg.above_z,
                grasping_min_z=self._cfg.grasping_min_z,
            ))

            # ④ 构建物体识别 → 任务切换映射
            self._object_mapper = self._build_object_mapper()
            logger.info(f"Arm-{arm_id}: 初始化完成 (外部输入模式 "
                        f"server={self._ext_server.url} "
                        f"任务规则={len(self._object_mapper.rules)})")
            return

        # ① 无相机模式：跳过摄像头与 GGCNN，仅测试线程链路
        if self._no_camera:
            logger.warning(f"Arm-{arm_id}: 无相机模式 — 使用虚拟深度图")
            self._camera = None
            self._ggcnn = None
            return

        # ① 初始化相机 (本地合成 或 仿真服务器取图)
        if self._use_sim_camera:
            logger.info(f"Arm-{arm_id}: 初始化仿真摄像头 (HTTP)...")
            self._camera = SimCamera(
                self._client,
                width=self._cfg.cam_width,
                height=self._cfg.cam_height,
            )
        else:
            objs = self._sim_cfg.get_camera_objects(arm_id)
            if not objs:
                objs = None  # 使用 BuiltinCamera 默认物块
            logger.info(f"Arm-{arm_id}: 初始化本地合成摄像头 "
                        f"(物块={0 if objs is None else len(objs)})...")
            self._camera = BuiltinCamera(
                width=self._cfg.cam_width,
                height=self._cfg.cam_height,
                objects=objs,
                seed=arm_id,  # 每臂不同种子，避免完全相同的画面
            )

        ci, _ = self._camera.get_intrinsics()
        self._K = np.array([
            [ci.fx, 0, ci.ppx],
            [0, ci.fy, ci.ppy],
            [0, 0, 1],
        ])

        # ② 加载 GGCNN2 模型
        model_path = self._cfg.model_file
        if not model_path:
            model_path = os.path.join(_demo_dir, 'models', 'epoch_50_cornell')
        logger.info(f"Arm-{arm_id}: 加载 GGCNN 模型 {model_path}...")
        self._ggcnn = TorchGGCNN({
            'MODEL_FILE': model_path,
            'OPEN_LOOP_HEIGHT': 0,
            'GGCNN_IN_THREAD': False,
            'DEPTH_CAM_K': self._K,
        }, Queue(1), Queue(1))
        time.sleep(0.5)

        # ③ 构建 TaskBuilder (使用该臂的观察位/释放位)
        self._builder = TaskBuilder(SimGraspConfig(
            detect_xyz=list(self._cfg.detect_xyz),
            release_xyz=list(self._cfg.release_xyz),
            above_z=self._cfg.above_z,
            grasping_min_z=self._cfg.grasping_min_z,
        ))

        # ④ 构建物体识别 → 任务切换映射
        self._object_mapper = self._build_object_mapper()

        logger.info(f"Arm-{arm_id}: 初始化完成 "
                    f"(观察位={self._cfg.detect_xyz} 释放位={self._cfg.release_xyz} "
                    f"任务规则={len(self._object_mapper.rules)})")

    def _build_object_mapper(self) -> ObjectMapper:
        """从配置构建物体识别映射。

        当前实现: 基于坐标最近邻匹配 (ObjectMapper.recognize)。
        未来接入视觉识别时，只需替换 recognize() 内部实现，此处不变。
        """
        mapper = ObjectMapper()
        rules_cfg = self._sim_cfg.task_rules.get(self._cfg.arm_id, {})
        for obj_type, rule_dict in rules_cfg.items():
            r = dict(rule_dict)
            r['object_type'] = obj_type
            mapper.add_rule(task_rule_from_dict(r))
        if rules_cfg:
            logger.info(f"Arm-{self._cfg.arm_id}: 加载物体识别规则 "
                        f"{len(rules_cfg)} 条 → {sorted(rules_cfg.keys())}")
        return mapper

    # ═══════════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════════

    def _main_loop(self):
        """主事件循环 — 图像获取 → GGCNN推理 → 聚类锁定 → 发送仿真任务。"""
        cfg = self._cfg
        arm_id = cfg.arm_id

        cs = min(cfg.cam_height, cfg.cam_width)
        off_r = max(0, cfg.cam_height - cs) // 2
        off_c = max(0, cfg.cam_width - cs) // 2

        logger.info(f"Arm-{arm_id}: 进入主循环")

        while (self._running and not self._stop_event.is_set()):
            try:
                # ── 发布虚拟位置 (仿真中臂固定在观察位) ──
                eef = self._get_sim_eef_pose_m()
                eef_mm = [eef[0]*1000, eef[1]*1000, eef[2]*1000,
                          eef[3], eef[4], eef[5]]
                self._coord.update_arm_position(arm_id, tuple(eef_mm))

                # ── 感知：外部坐标输入 或 摄像头 GGCNN 二选一 ──
                goal = None
                stable = False
                ext_label = ""   # 外部输入携带的 label (object_type 提示)

                if self._use_ext_input:
                    # 外部坐标输入模式：轮询共享服务器，按 arm_id 取目标
                    target, label = None, None
                    result = self._ext_server.get_target_for(arm_id)
                    if result is not None:
                        target, label = result
                    if target is not None:
                        valid, corrected, reason = self._validate_external_target(
                            target
                        )
                        if not valid:
                            logger.warning(
                                f"Arm-{arm_id}: 外部坐标校验失败 — {reason} "
                                f"— 已丢弃: {target}"
                            )
                        else:
                            if corrected != target:
                                logger.info(
                                    f"Arm-{arm_id}: 外部坐标已修正: {reason} → "
                                    f"X={corrected[0]:.1f} Y={corrected[1]:.1f}"
                                )
                            goal = corrected
                            stable = True
                            ext_label = label or ""
                            self._cand_buf.clear()

                    # 外部输入模式的占位画面 (无相机/GGCNN)
                    color_image = np.zeros(
                        (cfg.cam_height, cfg.cam_width, 3), dtype=np.uint8
                    )
                    color_crop = color_image[off_r:off_r+cs, off_c:off_c+cs].copy()
                    cv2.putText(
                        color_crop, "MANUAL MODE", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                    )
                    if stable:
                        cv2.putText(
                            color_crop,
                            f"X={goal[0]:.0f} Y={goal[1]:.0f} "
                            f"Yaw={goal[5]:.0f} {ext_label or ''}",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1
                        )
                    else:
                        cv2.putText(
                            color_crop, "WAITING FOR INPUT", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 200, 255), 1
                        )
                    # heatmap 占位 (无 GGCNN)
                    grasp_img = np.zeros((cs, cs, 3), dtype=np.uint8)
                    cv2.putText(
                        grasp_img, "EXT INPUT", (cs // 5, cs // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                    )
                else:
                    # ── 摄像头模式：RGB-D + GGCNN 推理 ──
                    if self._no_camera:
                        # 无相机模式：虚拟深度图 (无物块)，仅测试线程链路
                        color_image = np.zeros(
                            (cfg.cam_height, cfg.cam_width, 3), dtype=np.uint8
                        )
                        depth_image = np.ones(
                            (cfg.cam_height, cfg.cam_width),
                            dtype=np.float32
                        ) * 0.5
                        grasp_img = np.zeros((cs, cs, 3), dtype=np.uint8)
                        result = None
                    else:
                        color_image, depth_image = self._camera.get_images(
                            align=True
                        )
                        depth_image = depth_image.astype(np.float32)
                        grasp_img, result = self._ggcnn.get_grasp_img(
                            depth_image, self._K, eef[2]
                        )

                    color_crop = color_image[off_r:off_r+cs, off_c:off_c+cs].copy()

                    # ── 候选缓冲 + 聚类分析 ──
                    if (result is not None
                            and result[2] > cfg.min_result_z_mm / 1000.0):
                        cand = self._cam_result_to_base(result)
                        self._cand_buf.append(cand)

                        if len(self._cand_buf) > cfg.cand_window:
                            self._cand_buf.pop(0)

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

                        mp = self._ggcnn.prev_mp
                        cv2.circle(
                            color_crop, (int(mp[1]), int(mp[0])), 6,
                            (0, 255, 0) if stable else (0, 200, 255), -1
                        )
                    else:
                        self._cand_buf.clear()

                # ── 更新 Coordinator 状态 ──
                if stable:
                    self._coord.update_arm_state(arm_id, ArmState.LOCKED)
                else:
                    self._coord.update_arm_state(arm_id, ArmState.IDLE)

                # ── 触发抓取 ──
                if (stable and goal is not None
                        and self._in_own_zone(goal)
                        and (time.monotonic() - self._last_grasp_time)
                        > cfg.cooldown_sec):

                    # 是否需要协调区权限
                    need_coord_zone = self._which_coord_zone(goal[:2])
                    if need_coord_zone and not self._coord.request_zone(
                        arm_id, need_coord_zone
                    ):
                        alt = self._snap_to_exclusive_zone(goal)
                        if alt is not None:
                            goal = alt
                            logger.info(
                                f"Arm-{arm_id}: 协调区忙，已微移到独占区"
                                f" X={goal[0]:.1f} Y={goal[1]:.1f}"
                            )
                        else:
                            logger.debug(
                                f"Arm-{arm_id}: 协调区忙，放弃本轮"
                            )
                            self._publish_frame(color_crop, grasp_img,
                                                "WAIT_ZONE")
                            continue

                    # 构建并发送任务序列
                    self._coord.update_arm_state(arm_id, ArmState.GRASPING)
                    object_type, release_xyz = self._recognize_and_select_task(
                        goal, label=ext_label or None
                    )
                    sequence = self._builder.build_pick_and_place(
                        list(goal), release_xyz=release_xyz
                    )
                    # 附带任务上下文供落盘记录 (object_type / goal / release_xyz)
                    success = self._coord.send_task(
                        arm_id, sequence,
                        meta={
                            "object_type": object_type or "",
                            "goal": [round(v, 2) for v in goal],
                            "release_xyz": (list(release_xyz)
                                            if release_xyz else None),
                        },
                    )

                    if success:
                        self._grasp_count += 1
                        self._coord.record_grasp(arm_id)
                        # 抓取成功后从场景移除该物块，下一轮转向其他物块
                        if self._camera is not None and object_type:
                            self._camera.remove_object(object_type)
                        logger.info(
                            f"Arm-{arm_id}: 抓取完成 [{self._grasp_count}] "
                            f"目标 X={goal[0]:.1f} Y={goal[1]:.1f} "
                            f"Yaw={goal[5]:.1f} "
                            f"类型={self._current_object_type or '-'}"
                        )

                    # 释放协调区
                    if need_coord_zone:
                        self._coord.release_zone(arm_id, need_coord_zone)

                    self._last_grasp_time = time.monotonic()
                    self._cand_buf.clear()
                    self._coord.update_arm_state(arm_id, ArmState.IDLE)

                # ── 状态栏渲染 ──
                if self._use_ext_input:
                    state_label = "LOCKED" if stable else "WAITING INPUT"
                else:
                    state_label = ("LOCKED" if stable
                                   else ("SEARCHING" if goal is not None
                                         else "NO OBJECT"))
                cv2.putText(
                    color_crop,
                    f"{cfg.name} | {state_label} | "
                    f"{self._current_object_type or '-'} | #{self._grasp_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )

                self._publish_frame(color_crop, grasp_img, state_label,
                                    self._current_object_type)

            except Exception:
                logger.exception(f"Arm-{arm_id}: 主循环异常")
                time.sleep(0.5)

        logger.info(f"Arm-{arm_id}: 主循环退出")

    # ═══════════════════════════════════════════════════════════════════
    # 坐标变换
    # ═══════════════════════════════════════════════════════════════════

    def _get_sim_eef_pose_m(self) -> List[float]:
        """返回该臂的模拟末端位姿 (观察位, 相机朝下)。

        仿真模式下无真实机械臂，使用观察位作为"虚拟末端位姿"。

        返回:
            [x, y, z, roll, pitch, yaw] — 米, 弧度
        """
        d = self._cfg.detect_xyz
        return [d[0]*0.001, d[1]*0.001, d[2]*0.001,
                math.pi, 0.0, 0.0]

    def _cam_result_to_base(self, result: List[float]) -> List[float]:
        """将 GGCNN 相机系抓取点转换到该臂基坐标系。

        变换链: Base ← EEF(观察位) ← ColorCamera(手眼标定) ← DepthCamera

        参数:
            result: [x, y, z, ang, width, depth_center] — 相机坐标系 (米)

        返回:
            [X, Y, Z, 180, 0, Yaw] — mm, 度 (基坐标系)
        """
        cfg = self._cfg
        x, y, z, ang = result[0], result[1], result[2], result[3]
        gp = [x, y, z, 0, 0, -1 * ang]

        eef = self._get_sim_eef_pose_m()
        mat = (euler2mat(eef)
               * euler2mat(cfg.euler_eef_to_color_opt)
               * euler2mat(cfg.euler_color_to_depth_opt))
        gp_base = convert_pose(gp, mat)

        if gp_base[5] < -np.pi:
            gp_base[5] += np.pi
        elif gp_base[5] > 0:
            gp_base[5] -= np.pi

        return [
            gp_base[0]*1000, gp_base[1]*1000,
            gp_base[2]*1000 + cfg.gripper_z_mm,
            180, 0, math.degrees(gp_base[5] + np.pi/2),
        ]

    # ═══════════════════════════════════════════════════════════════════
    # 区域检查
    # ═══════════════════════════════════════════════════════════════════

    def _in_own_zone(self, goal: List[float]) -> bool:
        """判断抓取目标是否在有效工作区内（独占区或可申请的协调区）。"""
        x, y = goal[0], goal[1]
        if point_in_zone(x, y, self._cfg.workspace_zone):
            return True
        for cz in self._cfg.coordination_zones:
            if point_in_zone(x, y, cz):
                return True
        return False

    def _which_coord_zone(self, xy: Tuple[float, float]) -> Optional[str]:
        """判断 (x, y) 在哪个协调区内，返回区域名称；不在协调区返回 None。"""
        for i, cz in enumerate(self._cfg.coordination_zones):
            if point_in_zone(xy[0], xy[1], cz):
                return f"coord_{self._cfg.arm_id}_{i}"
        return None

    def _snap_to_exclusive_zone(
            self, goal: List[float]) -> Optional[List[float]]:
        """当目标在协调区但无法获取权限时，尝试将抓取点微调到独占区内。"""
        cfg = self._cfg
        y = goal[1]
        zone = cfg.workspace_zone
        margin = 30.0

        if y < zone[2] and y > zone[2] - margin:
            snapped = list(goal)
            snapped[1] = zone[2] + 5
            return snapped
        if y > zone[3] and y < zone[3] + margin:
            snapped = list(goal)
            snapped[1] = zone[3] - 5
            return snapped
        return None

    # ═══════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════

    def _publish_frame(self, color: np.ndarray, heatmap: np.ndarray,
                       state_label: str, object_type: str = ""):
        """发布最新帧到缓冲区（供 SimVisualizer 线程读取）。"""
        with self._frame_lock:
            self._latest_color = color.copy() if color is not None else None
            self._latest_heatmap = heatmap.copy() if heatmap is not None else None
            self._latest_state_label = state_label
            self._current_object_type = object_type

    def _recognize_and_select_task(
            self, goal: List[float],
            label: Optional[str] = None) -> Tuple[Optional[str],
                                                  Optional[List[float]]]:
        """识别抓取目标对应的物体类型，并选择对应的释放位置。

        参数:
            goal:  基坐标系抓取目标 [X, Y, Z, Roll, Pitch, Yaw]
            label: 外部输入携带的物体类型提示 (object_type)，可为 None；
                   非 None 时优先按类型查表，未命中回退坐标最近邻

        返回:
            (object_type, release_xyz) — 未识别到返回 (None, None)，
            调用方使用臂默认释放位。

        未来接入视觉识别: 只需替换 ObjectMapper.recognize 的实现，
        此方法无需改动。
        """
        rule = None
        if label:
            rule = self._object_mapper.recognize_by_type(label)
            if rule is None:
                logger.warning(
                    f"Arm-{self._cfg.arm_id}: label '{label}' 未匹配任务规则，"
                    f"回退坐标最近邻"
                )
        if rule is None:
            rule = self._object_mapper.recognize(goal[0], goal[1])
        if rule is None:
            self._current_object_type = ""
            return None, None
        self._current_object_type = rule.object_type
        logger.info(
            f"Arm-{self._cfg.arm_id}: 识别到 {rule.label} "
            f"(object_type={rule.object_type}) "
            f"@ X={goal[0]:.1f} Y={goal[1]:.1f} → "
            f"释放 {rule.release_xyz or self._cfg.release_xyz}"
        )
        return rule.object_type, rule.release_xyz

    def _validate_external_target(
            self, target: List[float]) -> Tuple[bool, List[float], str]:
        """校验外部输入的基坐标系抓取目标 [X, Y, Z, Roll, Pitch, Yaw] (mm/度)。

        与 run_simulation.py 的 validate_grasp_target 对齐，但工作区/Z 限位
        使用该臂自身配置 (workspace_zone/coordination_zones/above_z/
        grasping_min_z)。

        返回:
            (is_valid, corrected_target, reason)
        """
        cfg = self._cfg

        if target is None or len(target) < 3:
            return False, target, "目标为空或缺少 X/Y/Z"

        try:
            x, y, z = float(target[0]), float(target[1]), float(target[2])
            roll = float(target[3]) if len(target) > 3 else 180.0
            pitch = float(target[4]) if len(target) > 4 else 0.0
            yaw = float(target[5]) if len(target) > 5 else 0.0
        except (TypeError, ValueError) as e:
            return False, target, f"坐标值无法转换为数字: {e}"

        for name, val in [('X', x), ('Y', y), ('Z', z),
                          ('Roll', roll), ('Pitch', pitch), ('Yaw', yaw)]:
            if math.isnan(val) or math.isinf(val):
                return False, target, f"{name} 为 NaN/无穷大"

        # XY 工作区 (独占区或协调区) — 与触发抓取的 _in_own_zone 一致
        if not self._in_own_zone([x, y, 0, 0, 0, 0]):
            return False, target, "XY 超出工作区 (独占区/协调区)"

        # Z 高度检查
        z_max = cfg.above_z + 100
        if z < cfg.grasping_min_z:
            return False, target, f"Z 低于最低限位 {cfg.grasping_min_z} mm"
        if z > z_max:
            return False, target, f"Z 高于最大允许高度 {z_max} mm"

        # 姿态校验与归一化 (Roll≈180° 朝下, Pitch≈0°, Yaw 钳位到 [-90,90])
        if not (170.0 <= roll <= 190.0):
            roll = 180.0
        if not (-15.0 <= pitch <= 15.0):
            pitch = 0.0
        yaw = yaw % 360.0
        if yaw > 180.0:
            yaw -= 360.0
        elif yaw < -180.0:
            yaw += 360.0
        if abs(yaw) > 90.0:
            yaw = max(-90.0, min(90.0, yaw))

        corrected = [x, y, z, roll, pitch, yaw]
        return True, corrected, "OK"

    def _cleanup(self):
        """清理资源：停相机、更新状态。"""
        arm_id = self._cfg.arm_id
        logger.info(f"Arm-{arm_id}: 开始清理...")
        self._coord.update_arm_state(arm_id, ArmState.STOPPED)
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        logger.info(f"Arm-{arm_id}: 清理完成")


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python sim_arm_controller.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("SimArmController 自测")
    print("=" * 60)

    from simulation.sim_3arm.sim_coordinator import load_sim_config

    cfg_path = os.path.join(_simulation_dir, 'config_sim_3arms.json')
    sim_cfg = load_sim_config(cfg_path)
    # 用不可达本地地址作为 fast-fail 服务器，避免自测被 POST 阻塞挂起
    client = SimulationClient("http://127.0.0.1:1", timeout=2.0, retries=0)
    coord = SimCoordinator(sim_cfg, client=client)

    # 创建第一个臂的控制器 (本地合成摄像头)
    arm_cfg = sim_cfg.arms[0]
    print(f"\n[1] 创建 Arm-{arm_cfg.arm_id} ({arm_cfg.name})...")
    ctrl = SimArmController(sim_cfg, arm_cfg, coord,
                            use_sim_camera=False)
    ctrl.start()

    # 运行数秒
    print("\n[2] 运行 5 秒...")
    time.sleep(5.0)

    # 验证帧已发布
    color, heatmap, label, obj_type, count = ctrl.get_latest_frame()
    print(f"    最新帧: color={None if color is None else color.shape} "
          f"heatmap={None if heatmap is None else heatmap.shape} "
          f"label={label} obj_type={obj_type} count={count}")
    assert color is not None and heatmap is not None, "帧未被发布"
    assert ctrl._object_mapper is not None, "ObjectMapper 未构建"
    assert len(ctrl._object_mapper.rules) == 3, \
        f"任务规则应为 3 条, 实际 {len(ctrl._object_mapper.rules)}"
    print(f"    任务规则: {[r.object_type for r in ctrl._object_mapper.rules]}")

    print("\n[3] 停止...")
    ctrl.stop()
    ctrl.join(timeout=5.0)
    assert not ctrl._thread.is_alive(), "线程未退出"

    coord.close()
    print("\n自测完成 [PASS]")
