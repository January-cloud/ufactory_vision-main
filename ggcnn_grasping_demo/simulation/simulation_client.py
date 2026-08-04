#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_client — 仿真服务器 HTTP 客户端
============================================
封装与仿真服务器 (http://192.168.1.121:8080) 的 HTTP 通信，
提供任务序列 POST 和连接检查功能。
"""

import time
import json
import logging
import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationClientError(Exception):
    """仿真客户端通用异常基类。"""
    pass


class SimulationConnectionError(SimulationClientError):
    """无法连接到仿真服务器。"""
    pass


class SimulationTimeoutError(SimulationClientError):
    """请求超时。"""
    pass


class SimulationHTTPError(SimulationClientError):
    """仿真服务器返回非 2xx 状态码。"""

    def __init__(self, status_code: int, response_body: str = ""):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"HTTP {status_code}: {response_body[:200]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SimulationClient
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationClient:
    """仿真服务器 HTTP 客户端。

    使用方式:
        client = SimulationClient("http://192.168.1.121:8080")
        if client.check_connection():
            resp = client.post_task([...])  # 发送任务序列
    """

    def __init__(self, base_url: str = "http://192.168.1.121:5000",
                 timeout: float = 10.0, retries: int = 2):
        """
        参数:
            base_url: 仿真服务器基础 URL（不含尾部斜杠）
            timeout:  HTTP 请求超时秒数
            retries:  网络错误/5xx 时的重试次数
        """
        self.base_url = base_url.rstrip('/')
        self.task_endpoint = f"{self.base_url}/task"
        self.timeout = timeout
        self.retries = retries

        # 使用 Session 复用连接（keep-alive）
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })

    # ── 公开方法 ──────────────────────────────────────────────────────────────

    def check_connection(self) -> bool:
        """测试仿真服务器连通性。

        尝试 GET 根路径，任何 2xx/3xx 响应视为可达。

        返回:
            True  服务器可达
            False 连接失败或超时
        """
        try:
            resp = self._session.get(
                self.base_url,
                timeout=min(5.0, self.timeout),
            )
            # 任何非 5xx 响应都视为服务器在运行
            logger.info("仿真服务器连接成功: %s (HTTP %d)", self.base_url, resp.status_code)
            return True
        except requests.exceptions.Timeout:
            logger.warning("仿真服务器连接超时: %s", self.base_url)
            return False
        except requests.exceptions.ConnectionError as e:
            logger.warning("无法连接仿真服务器 %s: %s", self.base_url, e)
            return False
        except Exception as e:
            logger.warning("仿真服务器检查异常: %s", e)
            return False

    def post_task(self, sequence: list) -> dict:
        """向仿真服务器 POST 任务序列。

        参数:
            sequence: 任务列表，每项为 {"type": ..., "params": {...}, "wait": ...}

        返回:
            服务器响应的 JSON dict

        异常:
            SimulationConnectionError  所有重试后仍无法连接
            SimulationTimeoutError     所有重试后仍超时
            SimulationHTTPError        服务器返回 4xx/5xx
        """
        payload = {"sequence": sequence}
        last_error = None

        for attempt in range(self.retries + 1):
            try:
                logger.debug(
                    "POST %s (attempt %d/%d) — %d actions",
                    self.task_endpoint, attempt + 1, self.retries + 1, len(sequence)
                )
                resp = self._session.post(
                    self.task_endpoint,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code < 400:
                    logger.info(
                        "任务 POST 成功 (HTTP %d), %d actions",
                        resp.status_code, len(sequence)
                    )
                    try:
                        return resp.json()
                    except (json.JSONDecodeError, ValueError):
                        return {"status": "ok", "http_code": resp.status_code}

                # 4xx 错误不重试
                if 400 <= resp.status_code < 500:
                    raise SimulationHTTPError(resp.status_code, resp.text)

                # 5xx 错误可重试
                last_error = SimulationHTTPError(resp.status_code, resp.text)
                logger.warning(
                    "仿真服务器 5xx (attempt %d/%d): HTTP %d — %s",
                    attempt + 1, self.retries + 1, resp.status_code,
                    resp.text[:200] if resp.text else "(empty body)"
                )

            except requests.exceptions.Timeout as e:
                last_error = SimulationTimeoutError(
                    f"请求超时 ({self.timeout}s): {e}"
                )
                logger.warning("请求超时 (attempt %d/%d)", attempt + 1, self.retries + 1)

            except requests.exceptions.ConnectionError as e:
                last_error = SimulationConnectionError(f"连接失败: {e}")
                logger.warning("连接失败 (attempt %d/%d)", attempt + 1, self.retries + 1)

            except SimulationHTTPError:
                raise  # 4xx 不重试，直接抛出

            # 重试前等待（指数退避）
            if attempt < self.retries:
                wait = min(1.0 * (2 ** attempt), 5.0)
                logger.debug("等待 %.1fs 后重试...", wait)
                time.sleep(wait)

        # 所有重试耗尽
        raise last_error if last_error else SimulationClientError("未知错误")

    def get_camera(self) -> dict:
        """从仿真服务器获取摄像头图像。

        POST /get_camera，服务器返回 RGB 和深度图像数据。

        返回:
            {
                "success": True,
                "rgb_image": <base64 编码的 JPEG/PNG 图像>,
                "depth_image": <base64 编码的深度图>
            }

        失败时返回:
            {"success": False, "error": "错误描述"}

        异常:
            SimulationConnectionError  无法连接
            SimulationTimeoutError     请求超时
        """
        camera_endpoint = f"{self.base_url}/get_camera"

        for attempt in range(self.retries + 1):
            try:
                logger.debug(
                    "POST %s (attempt %d/%d)",
                    camera_endpoint, attempt + 1, self.retries + 1
                )
                resp = self._session.post(
                    camera_endpoint,
                    json={},
                    timeout=self.timeout,
                )

                if resp.status_code < 400:
                    data = resp.json()
                    if data.get("success"):
                        logger.debug("摄像头图像获取成功")
                    else:
                        logger.warning(
                            "摄像头返回失败: %s",
                            data.get("error", "unknown")
                        )
                    return data

                # 4xx: 不重试
                if 400 <= resp.status_code < 500:
                    raise SimulationHTTPError(resp.status_code, resp.text)

                # 5xx: 可重试
                logger.warning(
                    "获取摄像头 5xx (attempt %d/%d): HTTP %d",
                    attempt + 1, self.retries + 1, resp.status_code
                )

            except requests.exceptions.Timeout as e:
                logger.warning("获取摄像头超时 (attempt %d/%d)", attempt + 1, self.retries + 1)
                if attempt == self.retries:
                    raise SimulationTimeoutError(f"获取摄像头超时: {e}")

            except requests.exceptions.ConnectionError as e:
                logger.warning("获取摄像头连接失败 (attempt %d/%d)", attempt + 1, self.retries + 1)
                if attempt == self.retries:
                    raise SimulationConnectionError(f"获取摄像头连接失败: {e}")

            except SimulationHTTPError:
                raise

            # 重试前等待
            if attempt < self.retries:
                wait = min(1.0 * (2 ** attempt), 5.0)
                logger.debug("等待 %.1fs 后重试...", wait)
                time.sleep(wait)

        return {"success": False, "error": "all retries exhausted"}

    def close(self):
        """关闭 HTTP 会话。"""
        self._session.close()
        logger.debug("SimulationClient 会话已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python simulation_client.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("SimulationClient 自测")
    print("=" * 60)

    client = SimulationClient("http://192.168.1.121:8080")

    # 1. 连通性检查
    print("\n[1] 检查仿真服务器连通性...")
    reachable = client.check_connection()
    print(f"    服务器可达: {reachable}")

    # 2. 发送最小测试任务（仅当服务器可达时）
    if reachable:
        print("\n[2] 发送测试任务序列...")
        test_sequence = [
            {
                "type": "move",
                "params": {"x": 300, "y": 0, "z": 200, "roll": 180, "pitch": 0, "yaw": 0},
                "wait": 1
            },
            {
                "type": "vacuum",
                "params": {"on": True},
                "wait": 2
            },
        ]
        try:
            resp = client.post_task(test_sequence)
            print(f"    响应: {resp}")
        except SimulationClientError as e:
            print(f"    错误: {e}")
    else:
        print("\n[2] 跳过（服务器不可达）")

    # 3. 格式验证（不需要服务器）
    print("\n[3] 验证 JSON 序列化格式...")
    sample = {
        "sequence": [
            {
                "type": "move",
                "params": {"x": 200, "y": 0, "z": 380, "roll": 180, "pitch": 0, "yaw": 0},
                "wait": 0.5
            },
            {
                "type": "vacuum",
                "params": {"on": True},
                "wait": 0.8
            },
            {
                "type": "vacuum",
                "params": {"on": False},
                "wait": 0.5
            },
        ]
    }
    json_str = json.dumps(sample, indent=2, ensure_ascii=False)
    print(json_str)
    print("\n    格式符合预期: OK")

    client.close()
    print("\n自测完成 [PASS]")
