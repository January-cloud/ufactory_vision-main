#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_recorder — 生产任务落盘记录器
====================================
将三臂仿真中每次发送的生产任务持久化到磁盘，便于事后统计、回放与交接。

格式:
    JSON Lines (*.jsonl) — 每行一条任务记录，保留完整嵌套结构，
    包括 10 步 pick-and-place 动作序列。可直接用 pandas / jq / jsonl
    工具读取。

线程安全:
    TaskRecorder 内部使用锁保护写入，多臂控制线程可安全并发调用
    record()。每写一条立即 flush()，即使进程异常退出也不丢数据。

用法:
    recorder = TaskRecorder("logs/tasks.jsonl")
    recorder.record({
        "arm_id": 0, "arm_name": "Arm-Left",
        "object_type": "part_a", "goal": [...], "release_xyz": [...],
        "success": True, "sequence": [... 10 步动作 ...],
    })
    ...
    recorder.close()

数据流 (三臂仿真):
    SimArmController._main_loop
        → coordinator.send_task(arm_id, sequence, meta)
        → TaskRecorder.record({arm_id, arm_name, success, steps,
                               **meta, sequence, seq, timestamp})
"""

import os
import json
import time
import threading
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TaskRecorder:
    """生产任务落盘记录器 — 以 JSON Lines 格式追加写入任务记录。

    属性:
        path:   记录文件路径
        count:  本会话已记录的任务条数
        next_seq: 下一条记录将使用的序号
    """

    def __init__(self, path: str):
        """
        参数:
            path: 记录文件路径 (.jsonl)。父目录不存在时自动创建；
                  文件已存在时以追加模式写入。
        """
        parent = os.path.dirname(os.path.abspath(path)) or '.'
        os.makedirs(parent, exist_ok=True)

        self._path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._count = 0
        # 追加 + UTF-8 打开；每记录一条立即 flush，保证崩溃不丢数据
        self._file = open(path, 'a', encoding='utf-8')

        logger.info("TaskRecorder 就绪: %s", path)

    # ── 公开接口 ──

    def record(self, task: Dict[str, Any]) -> int:
        """追加一条任务记录。

        参数:
            task: 任务信息 dict。自动补充 seq 与 timestamp 字段
                  （已在 task 中给出时以 task 为准）。

        返回:
            本次记录的序号 (seq)
        """
        with self._lock:
            self._seq += 1
            entry = dict(task)
            entry.setdefault('seq', self._seq)
            entry.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
            self._file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self._file.flush()
            self._count += 1
            logger.debug("任务落盘 #%d (%s)", self._seq,
                         entry.get('object_type', '-'))
            return self._seq

    def close(self):
        """关闭记录文件。"""
        with self._lock:
            if not self._file.closed:
                self._file.close()

    # ── 只读属性 ──

    @property
    def path(self) -> str:
        return self._path

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def next_seq(self) -> int:
        with self._lock:
            return self._seq + 1


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python task_recorder.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import tempfile
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("TaskRecorder 自测")
    print("=" * 60)

    tmp_path = os.path.join(tempfile.mkdtemp(), 'tasks_selfcheck.jsonl')

    # 模拟一条真实的三臂任务记录 (字段结构与 sim_coordinator.send_task 一致)
    sample_task = {
        "arm_id": 0,
        "arm_name": "Arm-Left",
        "success": True,
        "steps": 10,
        "object_type": "part_a",
        "goal": [280.7, -149.8, 85.0, 180.0, 0.0, 30.0],
        "release_xyz": [150.0, -420.0, 83.0],
        "sequence": [
            {"type": "move", "params": {"x": 280.7, "y": -149.8, "z": 300.0},
             "wait": 0.5},
            {"type": "vacuum", "params": {"on": True}, "wait": 0.8},
        ],
    }

    recorder = TaskRecorder(tmp_path)
    print(f"\n[1] 记录 3 条 (含多线程并发)...")
    seqs = [recorder.record(sample_task) for _ in range(3)]
    assert seqs == [1, 2, 3], f"序号应连续: {seqs}"
    assert recorder.count == 3
    assert recorder.next_seq == 4
    print(f"    序号: {seqs}  count={recorder.count}  [PASS]")

    # 多线程并发写入
    print(f"\n[2] 4 线程并发写入各 5 条...")
    n_threads, n_each = 4, 5
    errors = []

    def _worker():
        try:
            for _ in range(n_each):
                recorder.record(sample_task)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"并发写入异常: {errors}"
    print(f"    并发写入后 count={recorder.count}  [PASS]")

    recorder.close()

    # 重新读取验证行数 + JSON 可解析 + seq 单调
    print(f"\n[3] 读取验证...")
    with open(tmp_path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f if ln.strip()]
    objs = [json.loads(ln) for ln in lines]
    seqs_on_disk = [o['seq'] for o in objs]
    assert len(objs) == 3 + n_threads * n_each, \
        f"行数不符: {len(objs)}"
    assert seqs_on_disk == sorted(seqs_on_disk), "seq 应单调递增"
    assert all(o['object_type'] == 'part_a' for o in objs)
    assert all(o['timestamp'] for o in objs), "每条都应有 timestamp"
    print(f"    共 {len(objs)} 行，seq 单调递增，字段完整  [PASS]")

    print(f"\n[4] 追加续写...")
    recorder2 = TaskRecorder(tmp_path)
    recorder2.record(sample_task)
    recorder2.close()
    with open(tmp_path, 'r', encoding='utf-8') as f:
        total = sum(1 for ln in f if ln.strip())
    assert total == len(objs) + 1, f"追加后应为 {len(objs)+1} 行, 实际 {total}"
    print(f"    追加后共 {total} 行  [PASS]")

    print(f"\n自测完成 [PASS]  → 临时文件: {tmp_path}")
