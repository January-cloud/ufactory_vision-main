#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_input — 外部坐标输入接收服务
======================================
轻量级 HTTP 服务器，在后台守护线程中运行，接收外部系统发来的
抓取目标坐标。接收到的坐标通过线程安全队列传递给主仿真循环。

端点:
  POST /grasp_target  — 接收单个抓取目标坐标
  GET  /status        — 服务器健康检查 + 队列状态

使用方式:
    from external_input import ExternalInputServer

    server = ExternalInputServer(host='0.0.0.0', port=8090)
    server.start()

    # 在主循环中非阻塞轮询
    target = server.get_target()
    if target is not None:
        # 使用外部坐标直接构建任务...
        pass

    server.stop()
"""

import json
import time
import queue
import socket
import threading
import logging
from typing import Optional, List, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP 请求处理器
# ═══════════════════════════════════════════════════════════════════════════════

class _GraspTargetHandler(BaseHTTPRequestHandler):
    """内部 HTTP 请求处理器 — 处理 /grasp_target 和 /status 请求。

    通过类属性 _shared_queue 与外部服务器实例共享队列。
    """

    # 类属性：由 ExternalInputServer 在创建时设置
    _shared_queue: Optional[queue.Queue] = None
    _server_start_time: float = 0.0

    def log_message(self, format, *args):
        """将默认的 stderr 日志重定向到 logging.debug，避免刷屏。"""
        logger.debug("HTTP: %s", format % args)

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        """发送 JSON 响应。"""
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """GET / 或 GET /status → 健康检查。"""
        if self.path in ('/', '/status'):
            qsize = 0
            if self._shared_queue is not None:
                qsize = self._shared_queue.qsize()
            uptime = time.monotonic() - self._server_start_time
            self._send_json(200, {
                'status': 'ok',
                'service': 'ExternalInputServer',
                'queue_size': qsize,
                'uptime_s': round(uptime, 1),
            })
        else:
            self._send_json(404, {'status': 'error', 'message': 'Not found'})

    def do_POST(self):
        """POST /grasp_target → 接收抓取目标坐标。"""
        if self.path != '/grasp_target':
            self._send_json(404, {'status': 'error', 'message': 'Not found'})
            return

        # 读取请求体
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {
                    'status': 'error',
                    'message': 'Empty request body'
                })
                return

            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode('utf-8'))
        except json.JSONDecodeError as e:
            self._send_json(400, {
                'status': 'error',
                'message': f'Invalid JSON: {e}'
            })
            return
        except Exception as e:
            self._send_json(400, {
                'status': 'error',
                'message': f'Bad request: {e}'
            })
            return

        # 验证必填字段
        try:
            x = float(body.get('x'))
            y = float(body.get('y'))
            z = float(body.get('z'))
        except (TypeError, ValueError):
            self._send_json(400, {
                'status': 'error',
                'message': 'Missing or invalid required fields: x, y, z (must be numeric)'
            })
            return

        # 可选字段（姿态，带默认值）
        roll = float(body.get('roll', 180.0))
        pitch = float(body.get('pitch', 0.0))
        yaw = float(body.get('yaw', 0.0))
        label = body.get('label', None)

        target = [x, y, z, roll, pitch, yaw]

        # 放入队列
        if self._shared_queue is not None:
            try:
                self._shared_queue.put_nowait(target)
                qsize = self._shared_queue.qsize()
                logger.info(
                    "外部坐标已接收: X=%.1f Y=%.1f Z=%.1f Yaw=%.1f (队列: %d)",
                    x, y, z, yaw, qsize
                )
            except queue.Full:
                # 队列满 → 丢弃最旧的目标，放入新的
                try:
                    self._shared_queue.get_nowait()
                    self._shared_queue.put_nowait(target)
                    logger.warning("外部坐标队列已满，已丢弃最旧目标")
                    self._send_json(200, {
                        'status': 'ok',
                        'warning': 'Queue was full, oldest target discarded',
                        'queued': self._shared_queue.qsize(),
                        'target': target,
                    })
                    return
                except queue.Empty:
                    pass

                self._send_json(503, {
                    'status': 'error',
                    'message': 'Queue full, cannot accept target'
                })
                return
        else:
            logger.error("共享队列未初始化!")
            self._send_json(500, {
                'status': 'error',
                'message': 'Internal server error: queue not initialized'
            })
            return

        resp = {
            'status': 'ok',
            'received': {
                'x': x, 'y': y, 'z': z,
                'roll': roll, 'pitch': pitch, 'yaw': yaw,
            },
            'queued': self._shared_queue.qsize(),
        }
        if label:
            resp['received']['label'] = label

        self._send_json(200, resp)


# ═══════════════════════════════════════════════════════════════════════════════
# ExternalInputServer
# ═══════════════════════════════════════════════════════════════════════════════

class ExternalInputServer:
    """外部坐标输入 HTTP 服务器。

    在后台守护线程中运行轻量级 HTTP 服务器，接收外部系统 POST 的
    抓取目标坐标 [X, Y, Z, Roll, Pitch, Yaw]（mm/度），通过线程安全
    队列传递给主仿真循环。

    使用方式:
        server = ExternalInputServer(host='0.0.0.0', port=8090)
        server.start()

        # 主循环中
        target = server.get_target()  # None 如果队列为空
        if target is not None:
            process(target)

        server.stop()
    """

    DEFAULT_PORT = 8090
    DEFAULT_MAX_QUEUE = 100

    def __init__(self, host: str = '0.0.0.0', port: int = DEFAULT_PORT,
                 max_queue: int = DEFAULT_MAX_QUEUE):
        """
        参数:
            host:      绑定地址 ('0.0.0.0' = 所有接口)
            port:      监听端口
            max_queue: 坐标队列最大容量（超过时丢弃最旧的）
        """
        self.host = host
        self.port = port
        self._max_queue = max_queue
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 将队列注入到处理器类属性中
        _GraspTargetHandler._shared_queue = self._queue
        _GraspTargetHandler._server_start_time = 0.0

    # ── 属性 ────────────────────────────────────────────────────────────────

    @property
    def url(self) -> str:
        """服务器完整 URL。"""
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """服务器是否正在运行。"""
        return self._running

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def start(self):
        """启动 HTTP 服务器（后台守护线程）。"""
        if self._running:
            logger.warning("ExternalInputServer 已在运行中")
            return

        try:
            self._server = HTTPServer((self.host, self.port), _GraspTargetHandler)
        except OSError as e:
            logger.error("无法绑定 %s:%d — %s", self.host, self.port, e)
            raise

        _GraspTargetHandler._server_start_time = time.monotonic()
        self._running = True
        self._thread = threading.Thread(
            target=self._serve_forever,
            name="ExternalInputServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ExternalInputServer 已启动: %s (POST /grasp_target, GET /status)",
            self.url
        )

    def stop(self):
        """停止 HTTP 服务器。"""
        if not self._running:
            return
        self._running = False

        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        logger.info("ExternalInputServer 已停止")

    def _serve_forever(self):
        """HTTP 服务器主循环（在守护线程中运行）。"""
        try:
            self._server.serve_forever(poll_interval=0.5)
        except Exception:
            if self._running:
                logger.exception("ExternalInputServer 异常退出")

    # ── 目标获取 ────────────────────────────────────────────────────────────

    def get_target(self) -> Optional[List[float]]:
        """非阻塞获取最新外部坐标。

        一次性清空队列中的所有目标，只返回最新的（丢弃中间过时的坐标）。
        这样外部系统可以快速连续发送坐标，而系统始终使用最新值。

        返回:
            [X, Y, Z, Roll, Pitch, Yaw] 列表 (mm/度)，或 None 如果队列为空
        """
        latest = None
        drained = 0
        while True:
            try:
                latest = self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break

        if drained > 1:
            logger.debug("已丢弃 %d 个过时目标，使用最新值", drained - 1)

        return latest

    def has_target(self) -> bool:
        """检查是否有外部坐标等待处理（不取出）。"""
        return not self._queue.empty()

    @property
    def queue_size(self) -> int:
        """当前队列中的目标数量。"""
        return self._queue.qsize()

    # ── 上下文管理器 ──────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python external_input.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    print("=" * 60)
    print("ExternalInputServer 自测")
    print("=" * 60)

    # 1. 启动服务器
    print("\n[1] 启动服务器...")
    server = ExternalInputServer(host='127.0.0.1', port=18090)
    server.start()
    print(f"    服务器 URL: {server.url}")
    print(f"    正在运行: {server.is_running}")
    assert server.is_running
    time.sleep(0.3)  # 等待线程启动
    print("    [PASS] 服务器启动")

    # 2. 队列为空时 get_target() 返回 None
    print("\n[2] 空队列测试...")
    result = server.get_target()
    print(f"    get_target() = {result}")
    assert result is None
    assert not server.has_target()
    print("    [PASS] 空队列")

    # 3. 通过 HTTP POST 发送测试坐标
    print("\n[3] HTTP POST /grasp_target 测试...")
    import urllib.request

    test_data = json.dumps({
        "x": 230.0,
        "y": -50.0,
        "z": 85.0,
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": -30.0,
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{server.url}/grasp_target",
        data=test_data,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp_data = json.loads(resp.read().decode('utf-8'))
            print(f"    响应: {json.dumps(resp_data, indent=4)}")
            assert resp_data['status'] == 'ok'
            assert resp_data['received']['x'] == 230.0
            assert resp_data['received']['yaw'] == -30.0
            print("    [PASS] POST 成功")
    except Exception as e:
        print(f"    错误: {e}")
        server.stop()
        raise

    # 4. 从队列取出坐标
    print("\n[4] 队列取出测试...")
    time.sleep(0.1)
    assert server.has_target()
    target = server.get_target()
    print(f"    取出的目标: {target}")
    assert target is not None
    assert target[0] == 230.0
    assert target[1] == -50.0
    assert target[2] == 85.0
    assert target[3] == 180.0  # roll
    assert target[4] == 0.0    # pitch
    assert target[5] == -30.0  # yaw
    assert not server.has_target()
    print("    [PASS] 队列取出")

    # 5. 多个目标 → 只返回最新的
    print("\n[5] 多目标去重测试...")
    for i in range(5):
        data = json.dumps({"x": 100.0 + i, "y": 0.0, "z": 50.0}).encode('utf-8')
        req = urllib.request.Request(
            f"{server.url}/grasp_target",
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        urllib.request.urlopen(req, timeout=5).close()

    time.sleep(0.1)
    target = server.get_target()
    print(f"    返回的目标: X={target[0]}")
    assert target[0] == 104.0, f"Expected X=104 (latest), got {target[0]}"
    print("    [PASS] 最新目标优先")

    # 6. GET /status 测试
    print("\n[6] GET /status 测试...")
    req = urllib.request.Request(f"{server.url}/status")
    with urllib.request.urlopen(req, timeout=5) as resp:
        status = json.loads(resp.read().decode('utf-8'))
        print(f"    状态: {json.dumps(status, indent=4)}")
        assert status['status'] == 'ok'
        assert status['service'] == 'ExternalInputServer'
        assert 'queue_size' in status
        assert 'uptime_s' in status
        print("    [PASS] 状态检查")

    # 7. 无效 JSON 错误处理
    print("\n[7] 无效请求错误处理...")
    # 7a. 无效 JSON
    req = urllib.request.Request(
        f"{server.url}/grasp_target",
        data=b'not json',
        headers={'Content-Type': 'application/json'},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        assert False, "Should have raised HTTPError"
    except urllib.error.HTTPError as e:
        print(f"    无效 JSON → HTTP {e.code}")
        assert e.code == 400
        print("    [PASS] 无效 JSON 返回 400")

    # 7b. 缺少必填字段
    req = urllib.request.Request(
        f"{server.url}/grasp_target",
        data=json.dumps({"x": 100}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        assert False, "Should have raised HTTPError"
    except urllib.error.HTTPError as e:
        print(f"    缺少 y,z → HTTP {e.code}")
        assert e.code == 400
        print("    [PASS] 缺少字段返回 400")

    # 8. 停止服务器
    print("\n[8] 停止服务器...")
    server.stop()
    assert not server.is_running
    time.sleep(0.2)
    # 确认端口已释放
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('127.0.0.1', 18090))
    s.close()
    assert result != 0, "Port should be free after stop"
    print("    [PASS] 服务器已停止，端口已释放")

    print(f"\n自测完成 [PASS]")
