#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coordinator — 中心协调器
=========================
多臂协同的核心模块。负责:
  - 臂注册与状态同步
  - 区域锁管理（租约 + 续约 + 自动回收）
  - 安全距离持续监控
  - 死锁检测与自动打破
  - HAZARD 区域管理
  - 紧急广播停止

线程安全：所有共享可变状态通过 threading.Lock + Condition 保护。
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

from .config import SystemConfig, ArmConfig, ArmState, ZoneState
from .collision_avoidance import point_in_zone, interarm_distance

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 区域锁租约
# ═══════════════════════════════════════════════════════════════════════════

class ZoneLease:
    """区域锁租约，带自动过期机制。"""

    __slots__ = ('arm_id', 'zone_name', 'granted_at',
                 'lease_duration', 'renewed_at')

    def __init__(self, arm_id: int, zone_name: str, lease_duration: float):
        self.arm_id = arm_id
        self.zone_name = zone_name
        self.granted_at = time.monotonic()
        self.lease_duration = lease_duration
        self.renewed_at = self.granted_at

    def renew(self):
        """续约，将到期时间延长到当前时刻 + lease_duration。"""
        self.renewed_at = time.monotonic()

    def is_expired(self) -> bool:
        """租约是否已过期。"""
        return (time.monotonic() - self.renewed_at) > self.lease_duration

    def remaining_s(self) -> float:
        """剩余有效时间 (秒)。"""
        return max(0.0, self.lease_duration -
                   (time.monotonic() - self.renewed_at))


# ═══════════════════════════════════════════════════════════════════════════
# 中心协调器
# ═══════════════════════════════════════════════════════════════════════════

class MultiArmCoordinator:
    """
    多臂中心协调器。

    使用方式:
        config = load_config("config_3arms.json")
        coord = MultiArmCoordinator(config)

        # 启动臂线程前必须先注册
        stop_evt = coord.register_arm(0)   # 返回该臂的停止信号 Event

        # 臂线程中持续调用:
        coord.update_arm_state(0, ArmState.IDLE)
        coord.update_arm_position(0, [x, y, z, r, p, yaw])

        # 请求进入协调区:
        if coord.request_zone(0, "coord_0_0"):
            # 在协调区内操作 ...
            coord.release_zone(0, "coord_0_0")

        # 每次安全移动前检查:
        if coord.is_safe_to_move(0, [x, y, z]):
            arm.set_position(...)
    """

    def __init__(self, config: SystemConfig):
        self._config = config
        # 使用可重入锁 (RLock)：get_summary() 在持锁状态下调用 any_hazard()
        # 等公开方法，普通 Lock 会导致同线程二次加锁死锁。
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        # ── 臂状态 ──
        self._arm_states: Dict[int, ArmState] = {}          # arm_id → 当前状态
        self._arm_positions: Dict[int, Tuple[float, ...]] = {}  # arm_id → 末端位姿
        self._arm_pos_timestamps: Dict[int, float] = {}     # arm_id → 位置更新时间
        self._arm_stop_events: Dict[int, threading.Event] = {}  # arm_id → 停止信号
        self._arm_registered: Dict[int, bool] = {}          # arm_id → 是否已注册

        # ── 区域管理 ──
        # zone_name → ZoneLease（仅 OWNED 状态的区域有租约）
        self._zone_leases: Dict[str, ZoneLease] = {}
        # zone_name → ZoneState
        self._zone_states: Dict[str, ZoneState] = {}

        # ── 等待队列 ──
        # zone_name → [arm_id, ...] 等待该区域的臂队列
        self._zone_wait_queues: Dict[str, List[int]] = {}
        # arm_id → zone_name 该臂正在等待哪个区域
        self._arm_waiting_for: Dict[int, str] = {}

        # ── 统计 ──
        self._grasp_counts: Dict[int, int] = {}      # arm_id → 成功抓取次数
        self._last_grasp_times: Dict[int, float] = {} # arm_id → 上次抓取时刻

        # ── 全局停止 ──
        self._global_stop_event = threading.Event()
        self._safety_monitor_thread: Optional[threading.Thread] = None

        # 初始化所有区域状态并启动安全监控
        self._init_zones()
        self._start_safety_monitor()

    # ═══════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════

    def _init_zones(self):
        """从配置初始化所有区域（全部标记为 FREE）。"""
        for arm in self._config.arms:
            zone_name = f"exclusive_{arm.arm_id}"
            self._zone_states[zone_name] = ZoneState.FREE
            self._zone_wait_queues[zone_name] = []

            for i, cz in enumerate(arm.coordination_zones):
                cz_name = f"coord_{arm.arm_id}_{i}"
                self._zone_states[cz_name] = ZoneState.FREE
                self._zone_wait_queues[cz_name] = []

    def _start_safety_monitor(self):
        """启动后台安全监控守护线程。"""
        self._safety_monitor_thread = threading.Thread(
            target=self._safety_monitor_loop,
            name="SafetyMonitor",
            daemon=True,
        )
        self._safety_monitor_thread.start()

    # ═══════════════════════════════════════════════════════════════════
    # 臂注册
    # ═══════════════════════════════════════════════════════════════════

    def register_arm(self, arm_id: int) -> threading.Event:
        """注册一个臂。返回该臂的停止信号 Event 对象。"""
        with self._lock:
            self._arm_states[arm_id] = ArmState.IDLE
            self._arm_positions[arm_id] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._arm_pos_timestamps[arm_id] = 0.0
            self._arm_stop_events[arm_id] = threading.Event()
            self._arm_registered[arm_id] = True
            self._grasp_counts[arm_id] = 0
            self._last_grasp_times[arm_id] = 0.0
            self._arm_waiting_for[arm_id] = ""

            # 标记独占区属于该臂（独占区永久持有，永不过期）
            zone_name = f"exclusive_{arm_id}"
            self._zone_states[zone_name] = ZoneState.OWNED
            self._zone_leases[zone_name] = ZoneLease(
                arm_id, zone_name, float('inf')
            )
            logger.info(f"[Coordinator] Arm-{arm_id} 已注册")
        return self._arm_stop_events[arm_id]

    # ═══════════════════════════════════════════════════════════════════
    # 状态更新
    # ═══════════════════════════════════════════════════════════════════

    def update_arm_state(self, arm_id: int, state: ArmState):
        """臂线程调用此方法更新自身状态。"""
        with self._lock:
            old = self._arm_states.get(arm_id)
            self._arm_states[arm_id] = state
            if old != state:
                logger.debug(f"[Coordinator] Arm-{arm_id}: {old} → {state}")

            # 如果进入 RECOVERING / STOPPED / DISCONNECTED，自动释放所有协调区
            if state in (ArmState.RECOVERING, ArmState.STOPPED,
                         ArmState.DISCONNECTED):
                self._release_all_coord_zones_locked(arm_id)

            # 通知等待者（可能有区域被释放）
            self._condition.notify_all()

    def update_arm_position(self, arm_id: int, position: Tuple[float, ...]):
        """臂线程更新当前末端位姿 (x,y,z,roll,pitch,yaw) mm/deg。"""
        with self._lock:
            self._arm_positions[arm_id] = position
            self._arm_pos_timestamps[arm_id] = time.monotonic()

    def get_arm_state(self, arm_id: int) -> Optional[ArmState]:
        """查询臂当前状态。"""
        with self._lock:
            return self._arm_states.get(arm_id)

    def get_arm_position(self, arm_id: int) -> Optional[Tuple[float, ...]]:
        """查询臂当前位置。"""
        with self._lock:
            return self._arm_positions.get(arm_id)

    # ═══════════════════════════════════════════════════════════════════
    # 区域锁管理
    # ═══════════════════════════════════════════════════════════════════

    def request_zone(self, arm_id: int, zone_name: str,
                     timeout_s: Optional[float] = None) -> bool:
        """
        请求获取某个区域的所有权。

        返回 True 表示获取成功（或已持有）。False 表示超时失败。
        - 独占区：只有其注册 owner 可以获取，直接返回结果。
        - 协调区：如果 FREE → 获得；如果被他人持有 → 排队等待至超时。
        """
        if timeout_s is None:
            timeout_s = self._config.zone_request_timeout_s

        deadline = time.monotonic() + timeout_s

        with self._lock:
            # 快速路径：已经持有该区域
            if self._check_zone_owned_by_locked(zone_name, arm_id):
                return True

            # 加入等待队列
            if zone_name not in self._zone_wait_queues:
                self._zone_wait_queues[zone_name] = []
            self._zone_wait_queues[zone_name].append(arm_id)
            self._arm_waiting_for[arm_id] = zone_name

            logger.debug(f"[Coordinator] Arm-{arm_id} 等待区域 '{zone_name}'")

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 超时：从等待队列移除
                    self._remove_from_wait_queue_locked(arm_id, zone_name)
                    logger.info(
                        f"[Coordinator] Arm-{arm_id} 获取 '{zone_name}' 超时"
                    )
                    return False

                # 尝试获取锁
                state = self._zone_states.get(zone_name, ZoneState.FREE)
                holder = self._get_zone_holder_locked(zone_name)

                if state == ZoneState.HAZARD:
                    # HAZARD 区域永久不可用
                    self._remove_from_wait_queue_locked(arm_id, zone_name)
                    return False

                if state == ZoneState.FREE or holder == arm_id:
                    # 区域可用！获得所有权
                    self._remove_from_wait_queue_locked(arm_id, zone_name)
                    self._grant_zone_locked(arm_id, zone_name)
                    return True

                # 区域不可用，检查是否形成死锁
                if holder is not None:
                    if self._detect_deadlock_locked(arm_id, holder, zone_name):
                        logger.warning(
                            f"[Coordinator] 检测到死锁 Arm-{arm_id} ↔ Arm-{holder}"
                        )
                        # arm_id 较小的一方主动放弃，打破环路
                        if arm_id < holder:
                            self._remove_from_wait_queue_locked(
                                arm_id, zone_name
                            )
                            return False
                        # 否则继续等待（另一方会在下一轮检测中放弃）

                # 等待（被 notify 唤醒或自然超时）
                self._condition.wait(timeout=min(remaining, 1.0))

    def renew_zone(self, arm_id: int, zone_name: str):
        """续约区域锁（在持有期内定期调用以防止过期）。"""
        with self._lock:
            lease = self._zone_leases.get(zone_name)
            if lease is not None and lease.arm_id == arm_id:
                lease.renew()

    def release_zone(self, arm_id: int, zone_name: str):
        """释放单个区域锁。"""
        with self._lock:
            self._release_zone_locked(arm_id, zone_name)
            self._condition.notify_all()

    def release_all_zones(self, arm_id: int):
        """释放该臂持有的所有协调区锁。"""
        with self._lock:
            self._release_all_coord_zones_locked(arm_id)
            self._condition.notify_all()

    # ═══════════════════════════════════════════════════════════════════
    # 安全移动前检查
    # ═══════════════════════════════════════════════════════════════════

    def is_safe_to_move(self, arm_id: int,
                        target_xyz: Tuple[float, float, float]) -> bool:
        """
        检查 arm_id 移动到 target_xyz 是否安全。

        检查项目:
          1. 目标是否在其他臂的独占区内（且该臂不处于 IDLE 安全状态）
          2. 目标是否在协调区内（且不由当前臂持有）
          3. 目标与其他臂 EE 位置的距离是否 >= safety_radius
        """
        tx, ty, tz = target_xyz

        with self._lock:
            # 检查1：独占区冲突
            for other_cfg in self._config.arms:
                if other_cfg.arm_id == arm_id:
                    continue
                if point_in_zone(tx, ty, other_cfg.workspace_zone):
                    other_state = self._arm_states.get(
                        other_cfg.arm_id, ArmState.IDLE
                    )
                    # 对方不处于安全状态 → 危险
                    if other_state not in (ArmState.IDLE, ArmState.DISCONNECTED):
                        # 例外：目标在高位（过顶穿越）是安全的
                        if tz < self._config.safety_height:
                            return False

            # 检查2：协调区权限
            for arm_cfg in self._config.arms:
                for i, cz in enumerate(arm_cfg.coordination_zones):
                    if point_in_zone(tx, ty, cz):
                        cz_name = f"coord_{arm_cfg.arm_id}_{i}"
                        holder = self._get_zone_holder_locked(cz_name)
                        state = self._zone_states.get(
                            cz_name, ZoneState.FREE
                        )
                        if state == ZoneState.HAZARD:
                            return False
                        if (holder is not None and holder != arm_id
                                and tz < self._config.safety_height):
                            return False

            # 检查3：与其他臂 EE 的安全距离
            for other_id, other_pos in self._arm_positions.items():
                if other_id == arm_id:
                    continue
                if other_pos[0] == 0.0 and other_pos[1] == 0.0:
                    continue  # 尚未初始化，跳过
                dist = interarm_distance(
                    [tx, ty, tz], list(other_pos[:3])
                )
                if dist < self._config.safety_radius_mm:
                    return False

        return True

    # ═══════════════════════════════════════════════════════════════════
    # 死锁检测
    # ═══════════════════════════════════════════════════════════════════

    def _detect_deadlock_locked(self, arm_id: int, holder_id: int,
                                wanted_zone: str) -> bool:
        """
        检测 arm_id → holder_id 是否存在循环等待。

        arm_id 想要 holder_id 持有的 wanted_zone，
        检测 holder_id 是否正在等待 arm_id 持有的某个区域（或形成三臂环）。
        """
        holder_waits_for = self._arm_waiting_for.get(holder_id, "")
        if not holder_waits_for:
            return False

        # 直接环: A→B→A
        holder_wants_zone_holder = self._get_zone_holder_locked(
            holder_waits_for
        )
        if holder_wants_zone_holder == arm_id:
            return True

        # 三级链: A→B→C→A
        if holder_wants_zone_holder is not None:
            third_waits_for = self._arm_waiting_for.get(
                holder_wants_zone_holder, ""
            )
            if third_waits_for:
                third_holder = self._get_zone_holder_locked(third_waits_for)
                if third_holder == arm_id:
                    return True

        return False

    # ═══════════════════════════════════════════════════════════════════
    # 安全监控
    # ═══════════════════════════════════════════════════════════════════

    def _safety_monitor_loop(self):
        """后台守护线程：持续监测臂间距离 + 位置新鲜度 + EE 卡死。"""
        interval = 1.0 / self._config.safety_monitor_hz
        logger.info(
            f"[SafetyMonitor] 启动 (间隔 {interval:.2f}s)"
        )

        while not self._global_stop_event.is_set():
            try:
                self._global_stop_event.wait(timeout=interval)
                if self._global_stop_event.is_set():
                    break
                self._run_safety_checks()
            except Exception:
                logger.exception("[SafetyMonitor] 异常")

    def _run_safety_checks(self):
        """执行一轮安全检查。收集违规事件，锁外日志输出以避免死锁。"""
        violations = []  # (消息文本, 需停止的臂ID列表)

        with self._lock:
            active_ids = [
                aid for aid, s in self._arm_states.items()
                if s not in (ArmState.IDLE, ArmState.RECOVERING,
                             ArmState.STOPPED, ArmState.DISCONNECTED)
            ]

            # ① 安全距离检查
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    a_id, b_id = active_ids[i], active_ids[j]
                    pos_a = self._arm_positions.get(a_id)
                    pos_b = self._arm_positions.get(b_id)
                    if pos_a is None or pos_b is None:
                        continue
                    dist = interarm_distance(list(pos_a), list(pos_b))
                    if dist < self._config.safety_radius_mm:
                        msg = (f"[SAFETY] Arm-{a_id} ↔ Arm-{b_id} "
                               f"间距 {dist:.0f}mm < "
                               f"{self._config.safety_radius_mm}mm!")
                        violations.append((msg, [a_id, b_id]))

            now = time.monotonic()

            # ② 位置新鲜度检查（L2 断连检测）
            for arm_id, ts in list(self._arm_pos_timestamps.items()):
                if ts > 0 and (now - ts) > self._config.position_stale_s:
                    state = self._arm_states.get(arm_id)
                    if state not in (ArmState.DISCONNECTED, ArmState.STOPPED):
                        self._arm_states[arm_id] = ArmState.DISCONNECTED
                        self._release_all_coord_zones_locked(arm_id)
                        violations.append(
                            (f"[Coordinator] Arm-{arm_id} 位置数据过期 "
                             f"({now - ts:.1f}s) → DISCONNECTED", [])
                        )

            # ③ EE 卡死检测（L3）
            for arm_id in active_ids:
                pos = self._arm_positions.get(arm_id)
                ts = self._arm_pos_timestamps.get(arm_id, 0)
                if pos is None or ts == 0:
                    continue
                # 检查该臂是否在非观察位停留过久
                if (now - ts) > self._config.ee_stuck_s:
                    arm_cfg = self._config.get_arm(arm_id)
                    if arm_cfg is not None:
                        dx = abs(pos[0] - arm_cfg.detect_xyz[0])
                        dy = abs(pos[1] - arm_cfg.detect_xyz[1])
                        if dx > 20 or dy > 20:  # 不在观察位
                            self._mark_arm_zones_hazard_locked(arm_id)
                            violations.append(
                                (f"[SAFETY] Arm-{arm_id} EE "
                                 f"{self._config.ee_stuck_s}s 未移动"
                                 f" → HAZARD", [])
                            )

            # ④ 租约过期检查
            for zone_name, lease in list(self._zone_leases.items()):
                if zone_name.startswith("coord_") and lease.is_expired():
                    self._release_zone_locked(lease.arm_id, zone_name)
                    violations.append(
                        (f"[Coordinator] 区域 '{zone_name}' 租约过期，"
                         f"回收自 Arm-{lease.arm_id}", [])
                    )

            # 在锁内执行臂停止信号设置
            for _, stop_list in violations:
                for aid in stop_list:
                    if aid in self._arm_stop_events:
                        self._arm_stop_events[aid].set()

        # 锁外日志输出（避免 logging 模块内部锁与 coordinator 锁形成死锁）
        for msg, _ in violations:
            logger.critical(msg)

    # ═══════════════════════════════════════════════════════════════════
    # 抓取统计
    # ═══════════════════════════════════════════════════════════════════

    def record_grasp(self, arm_id: int):
        """记录一次成功抓取。"""
        with self._lock:
            self._grasp_counts[arm_id] = self._grasp_counts.get(arm_id, 0) + 1
            self._last_grasp_times[arm_id] = time.monotonic()

    def get_grasp_counts(self) -> Dict[int, int]:
        """获取各臂抓取计数。"""
        with self._lock:
            return dict(self._grasp_counts)

    def get_last_grasp_time(self, arm_id: int) -> float:
        """获取某臂上次抓取时刻。"""
        with self._lock:
            return self._last_grasp_times.get(arm_id, 0.0)

    def all_idle(self) -> bool:
        """所有臂是否都处于 IDLE 状态。"""
        with self._lock:
            return all(
                s == ArmState.IDLE
                for s in self._arm_states.values()
            )

    def any_hazard(self) -> bool:
        """是否存在 HAZARD 区域。"""
        with self._lock:
            return any(
                s == ZoneState.HAZARD
                for s in self._zone_states.values()
            )

    def clear_hazard(self):
        """操作员确认清除所有 HAZARD 区域（需人工确认安全后调用）。"""
        with self._lock:
            for name, state in list(self._zone_states.items()):
                if state == ZoneState.HAZARD:
                    self._zone_states[name] = ZoneState.FREE
                    logger.info(f"[Coordinator] HAZARD 已清除: '{name}'")

    # ═══════════════════════════════════════════════════════════════════
    # 停止与清理
    # ═══════════════════════════════════════════════════════════════════

    def broadcast_stop(self, source_arm_id: Optional[int] = None):
        """广播全局停止信号。"""
        # 先设置全局停止事件（线程安全，无需持锁），让安全监控线程立刻退出
        self._global_stop_event.set()
        with self._lock:
            for evt in self._arm_stop_events.values():
                evt.set()
        logger.critical(
            f"[Coordinator] 全局停止 (来源: Arm-{source_arm_id})"
        )

    @property
    def global_stop_event(self) -> threading.Event:
        """全局停止事件（可视化线程用于判断退出）。"""
        return self._global_stop_event

    def get_summary(self) -> dict:
        """返回当前系统摘要（供 visualizer 显示状态面板）。"""
        with self._lock:
            zone_info = {}
            for name, state in self._zone_states.items():
                holder = None
                lease = self._zone_leases.get(name)
                if lease:
                    holder = lease.arm_id
                zone_info[name] = {
                    'state': state.value,
                    'holder': holder,
                }

            return {
                'arm_states': {k: v.value for k, v in self._arm_states.items()},
                'arm_positions': {
                    k: list(v) for k, v in self._arm_positions.items()
                },
                'grasp_counts': dict(self._grasp_counts),
                'zones': zone_info,
                'any_hazard': self.any_hazard(),
            }

    # ═══════════════════════════════════════════════════════════════════
    # 内部方法（调用时必须持有 _lock）
    # ═══════════════════════════════════════════════════════════════════

    def _check_zone_owned_by_locked(self, zone_name: str,
                                    arm_id: int) -> bool:
        """检查 zone 是否已被 arm_id 持有（锁内调用）。"""
        lease = self._zone_leases.get(zone_name)
        if lease is None:
            return False
        return lease.arm_id == arm_id and not lease.is_expired()

    def _get_zone_holder_locked(self, zone_name: str) -> Optional[int]:
        """获取区域当前持有者 arm_id，过期视为无人持有（锁内调用）。"""
        lease = self._zone_leases.get(zone_name)
        if lease is None:
            return None
        if lease.is_expired():
            del self._zone_leases[zone_name]
            self._zone_states[zone_name] = ZoneState.FREE
            return None
        return lease.arm_id

    def _grant_zone_locked(self, arm_id: int, zone_name: str):
        """授予区域所有权（锁内调用）。"""
        self._zone_states[zone_name] = ZoneState.OWNED
        self._zone_leases[zone_name] = ZoneLease(
            arm_id, zone_name, self._config.lease_duration_s
        )
        logger.debug(f"[Coordinator] '{zone_name}' → Arm-{arm_id}")

    def _release_zone_locked(self, arm_id: int, zone_name: str):
        """释放一个区域（锁内调用）。"""
        lease = self._zone_leases.get(zone_name)
        if lease is None:
            return
        if lease.arm_id != arm_id:
            return  # 不是当前臂持有的

        del self._zone_leases[zone_name]
        # 独占区不改变状态（永久属于注册臂），协调区恢复 FREE
        if zone_name.startswith("coord_"):
            self._zone_states[zone_name] = ZoneState.FREE
        logger.debug(f"[Coordinator] '{zone_name}' 释放自 Arm-{arm_id}")

    def _release_all_coord_zones_locked(self, arm_id: int):
        """释放该臂持有的所有协调区锁（锁内调用）。"""
        for zone_name in list(self._zone_leases.keys()):
            if zone_name.startswith("coord_"):
                self._release_zone_locked(arm_id, zone_name)

    def _mark_arm_zones_hazard_locked(self, arm_id: int):
        """将该臂当前所在的协调区标记为 HAZARD（锁内调用）。"""
        pos = self._arm_positions.get(arm_id)
        if pos is None:
            return
        x, y = pos[0], pos[1]

        for name, state in self._zone_states.items():
            if state == ZoneState.OWNED:
                lease = self._zone_leases.get(name)
                if lease and lease.arm_id == arm_id:
                    # 确认该区域是协调区且臂确实在其中
                    for arm_cfg in self._config.arms:
                        for i, cz in enumerate(arm_cfg.coordination_zones):
                            cz_name = f"coord_{arm_cfg.arm_id}_{i}"
                            if cz_name == name and point_in_zone(x, y, cz):
                                self._zone_states[name] = ZoneState.HAZARD
                                logger.critical(
                                    f"[Coordinator] '{name}' → HAZARD"
                                )

    def _remove_from_wait_queue_locked(self, arm_id: int, zone_name: str):
        """从等待队列中移除（锁内调用）。"""
        q = self._zone_wait_queues.get(zone_name, [])
        if arm_id in q:
            q.remove(arm_id)
        self._arm_waiting_for[arm_id] = ""
