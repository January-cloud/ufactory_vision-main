#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_camera — 仿真摄像头适配器
==============================
通过 HTTP 从仿真服务器获取 RGB 和深度图像，提供与 RealSenseCamera
兼容的接口，可作为 drop-in 替换。

图像格式:
  - rgb_image:  base64 编码的 JPEG/PNG → 解码为 BGR uint8 ndarray
  - depth_image: base64 编码的 16-bit PNG 或 raw 浮点数据 → float32 ndarray (米)

使用方式:
    from simulation_client import SimulationClient
    from sim_camera import SimCamera

    client = SimulationClient("http://192.168.1.121:8080")
    cam = SimCamera(client, width=640, height=480)
    color, depth = cam.get_images()       # 与 RealSenseCamera 接口一致
    ci, di = cam.get_intrinsics()         # 获取内参
"""

import cv2
import math
import base64
import logging
import numpy as np
from typing import Tuple, Optional
from collections import namedtuple

logger = logging.getLogger(__name__)

# 模拟 pyrealsense2.intrinsics 对象的轻量 namedtuple
Intrinsics = namedtuple('Intrinsics', ['fx', 'fy', 'ppx', 'ppy', 'width', 'height'])


class SimCamera:
    """仿真摄像头 — 通过 HTTP 从仿真平台获取图像。

    提供与 RealSenseCamera 兼容的接口:
      - get_images(align=True) → (color_bgr, depth_float32)
      - get_intrinsics() → (color_intrin, depth_intrin)
      - stop() → 释放资源

    图像解码策略:
      - RGB: base64 JPEG/PNG → cv2.imdecode → BGR uint8
      - Depth: 优先尝试 16-bit PNG 解码 (mm → m)
               回退到 float32 raw bytes
    """

    # 默认内参（D435 典型值，可配置覆盖）
    DEFAULT_FX = 615.0
    DEFAULT_FY = 615.0

    def __init__(self, client, width: int = 640, height: int = 480,
                 fx: float = None, fy: float = None,
                 depth_scale: float = 0.001):
        """
        参数:
            client:      SimulationClient 实例（已连接仿真服务器）
            width:       图像宽度 (像素)
            height:      图像高度 (像素)
            fx, fy:      相机焦距 (像素)，None 则使用默认值
            depth_scale: 深度图解码时的缩放因子 (默认 0.001 = mm→m)
        """
        self._client = client
        self.width = width
        self.height = height
        self.depth_scale = depth_scale

        cx = width / 2.0
        cy = height / 2.0
        self._fx = fx if fx is not None else self.DEFAULT_FX
        self._fy = fy if fy is not None else self.DEFAULT_FY
        self._cx = cx
        self._cy = cy

        # 构造 intriniscs 对象
        self._color_intrin = Intrinsics(
            fx=self._fx, fy=self._fy,
            ppx=cx, ppy=cy,
            width=width, height=height,
        )
        self._depth_intrin = Intrinsics(
            fx=self._fx, fy=self._fy,
            ppx=cx, ppy=cy,
            width=width, height=height,
        )

        self._frame_count = 0
        logger.info(
            "SimCamera 初始化: %dx%d fx=%.1f fy=%.1f",
            width, height, self._fx, self._fy
        )

    # ── 公开接口（与 RealSenseCamera 兼容）──────────────────────────────────

    def get_images(self, align: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """从仿真服务器获取一帧彩色图和深度图。

        参数:
            align: 忽略（仿真服务器返回的已是配准图像）

        返回:
            (color_bgr, depth_float32)
            color_bgr: uint8 ndarray (H, W, 3) BGR 格式
            depth_float32: float32 ndarray (H, W) 单位米，0 或 NaN 表示无效
        """
        data = self._client.get_camera()

        if not data.get("success"):
            error_msg = data.get("error", "unknown")
            logger.warning("仿真摄像头获取失败: %s", error_msg)
            raise RuntimeError(f"SimCamera: 获取图像失败 — {error_msg}")

        rgb = self._decode_rgb(data.get("rgb_image"))
        depth = self._decode_depth(data.get("depth_image"))

        self._frame_count += 1
        return rgb, depth

    def get_intrinsics(self, align: bool = False):
        """获取相机内参。

        返回:
            (color_intrin, depth_intrin)
            每个都是包含 fx, fy, ppx, ppy, width, height 的对象
        """
        return self._color_intrin, self._depth_intrin

    def stop(self):
        """释放资源（HTTP 摄像头无需特殊清理）。"""
        logger.info("SimCamera 停止 (共 %d 帧)", self._frame_count)
        self._frame_count = 0

    # ── 内参更新 ────────────────────────────────────────────────────────────

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """手动设置相机内参（如果仿真服务器未提供内参信息）。"""
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._color_intrin = Intrinsics(fx=fx, fy=fy, ppx=cx, ppy=cy,
                                        width=self.width, height=self.height)
        self._depth_intrin = Intrinsics(fx=fx, fy=fy, ppx=cx, ppy=cy,
                                        width=self.width, height=self.height)
        logger.info("内参已更新: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", fx, fy, cx, cy)

    # ── 图像解码（内部方法）────────────────────────────────────────────────

    def _decode_rgb(self, rgb_data) -> np.ndarray:
        """解码 RGB 图像数据为 BGR uint8 ndarray。

        支持格式:
          - base64 编码的 JPEG/PNG 字节串
          - None/空 → 返回全零图像
        """
        if rgb_data is None or (isinstance(rgb_data, str) and len(rgb_data) == 0):
            logger.warning("RGB 数据为空，返回默认图像")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        try:
            # base64 → bytes → OpenCV 解码
            if isinstance(rgb_data, str):
                img_bytes = base64.b64decode(rgb_data)
            elif isinstance(rgb_data, bytes):
                img_bytes = rgb_data
            else:
                logger.error("不支持的 RGB 数据类型: %s", type(rgb_data))
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("RGB 图像解码失败，返回默认图像")
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # 确保尺寸匹配（必要时 resize）
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = cv2.resize(img, (self.width, self.height))

            return img  # OpenCV 解码后已是 BGR

        except Exception as e:
            logger.error("RGB 解码异常: %s", e)
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _decode_depth(self, depth_data) -> np.ndarray:
        """解码深度图像数据为 float32 ndarray (米)。

        支持格式:
          - base64 编码的 16-bit PNG (mm 单位) → 缩放为米
          - base64 编码的 raw float32 字节
          - None/空 → 返回全 NaN 图像
        """
        if depth_data is None or (isinstance(depth_data, str) and len(depth_data) == 0):
            logger.warning("深度数据为空，返回 NaN 图像")
            return np.full((self.height, self.width), math.nan, dtype=np.float32)

        try:
            # base64 → bytes
            if isinstance(depth_data, str):
                img_bytes = base64.b64decode(depth_data)
            elif isinstance(depth_data, bytes):
                img_bytes = depth_data
            else:
                logger.error("不支持的深度数据类型: %s", type(depth_data))
                return np.full((self.height, self.width), math.nan, dtype=np.float32)

            # 尝试 PNG 解码（16-bit 深度图，mm 单位）
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            if img is not None:
                # PNG 解码成功 → 假设为 uint16 mm 单位
                depth = img.astype(np.float32) * self.depth_scale
            else:
                # 回退：尝试 raw float32 解码
                logger.debug("PNG 解码失败，尝试 raw float32 解码")
                if len(img_bytes) >= self.width * self.height * 4:
                    depth = np.frombuffer(
                        img_bytes, dtype=np.float32
                    ).reshape((self.height, self.width)).copy()
                else:
                    logger.error("无法解码深度数据 (%d bytes)", len(img_bytes))
                    return np.full((self.height, self.width), math.nan, dtype=np.float32)

            # 确保尺寸匹配
            if depth.shape[0] != self.height or depth.shape[1] != self.width:
                depth = cv2.resize(depth, (self.width, self.height))

            # 0 值替换为 NaN（模拟 RealSense 的无效区域）
            depth[depth <= 0] = math.nan

            return depth.astype(np.float32)

        except Exception as e:
            logger.error("深度解码异常: %s", e)
            return np.full((self.height, self.width), math.nan, dtype=np.float32)

    # ── 属性 ────────────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """已获取的帧数。"""
        return self._frame_count


# ═══════════════════════════════════════════════════════════════════════════════
# 自测（直接运行 python sim_camera.py）
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    print("=" * 60)
    print("SimCamera 自测")
    print("=" * 60)

    # 1. 测试 base64 编解码（不依赖服务器）
    print("\n[1] RGB base64 编解码测试...")
    # 创建测试图像并编码
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', test_img)
    b64_str = base64.b64encode(encoded.tobytes()).decode('utf-8')

    # 用 SimCamera 的内部解码器验证
    from simulation_client import SimulationClient
    client = SimulationClient("http://192.168.1.121:5000")
    cam = SimCamera(client)
    decoded = cam._decode_rgb(b64_str)
    print(f"    原始: {test_img.shape} {test_img.dtype}")
    print(f"    解码: {decoded.shape} {decoded.dtype}")
    assert decoded.shape == (480, 640, 3), "RGB shape mismatch"
    print("    [PASS] RGB 编解码")

    # 2. 测试深度图编解码
    print("\n[2] Depth base64 编解码测试...")
    test_depth_mm = np.random.randint(1, 5000, (480, 640), dtype=np.uint16)
    _, encoded_d = cv2.imencode('.png', test_depth_mm)
    b64_depth = base64.b64encode(encoded_d.tobytes()).decode('utf-8')

    decoded_depth = cam._decode_depth(b64_depth)
    print(f"    原始: {test_depth_mm.shape} {test_depth_mm.dtype} (mm)")
    print(f"    解码: {decoded_depth.shape} {decoded_depth.dtype} (m)")
    print(f"    值范围: [{np.nanmin(decoded_depth):.3f}, {np.nanmax(decoded_depth):.3f}]")
    assert decoded_depth.shape == (480, 640), "Depth shape mismatch"
    assert decoded_depth.dtype == np.float32, "Depth dtype should be float32"
    print("    [PASS] Depth 编解码")

    # 3. 测试 get_intrinsics
    print("\n[3] 内参测试...")
    ci, di = cam.get_intrinsics()
    print(f"    Color: fx={ci.fx} fy={ci.fy} ppx={ci.ppx} ppy={ci.ppy} {ci.width}x{ci.height}")
    print(f"    Depth: fx={di.fx} fy={di.fy} ppx={di.ppx} ppy={di.ppy} {di.width}x{di.height}")
    print("    [PASS] 内参")

    # 4. 测试空数据处理
    print("\n[4] 空数据容错测试...")
    empty_rgb = cam._decode_rgb(None)
    empty_depth = cam._decode_depth(None)
    print(f"    空 RGB:  {empty_rgb.shape} (全零={np.all(empty_rgb==0)})")
    print(f"    空 Depth: {empty_depth.shape} (全NaN={np.all(np.isnan(empty_depth))})")
    print("    [PASS] 空数据容错")

    cam.stop()
    client.close()
    print(f"\n自测完成 [PASS]")
