# ufactory_vision 项目目录结构说明

> 基于 GGCNN + RealSense D435 + xArm Lite6 的视觉引导机器人抓取系统
> 含三臂协同扩展（multi_arm 包）和仿真平台接口（simulation 包）

---

## 一、项目根目录

```
ufactory_vision-main/
├── README.md                          # 项目英文说明文档
├── README_ZH.md                       # 项目中文说明文档
├── LICENSE                            # BSD 3-Clause 许可证 (UFACTORY)
├── .gitignore                         # Git 忽略规则
│
├── assets/                            # 硬件参考图片
│   ├── realsense_d435.jpg             # Intel D435 相机照片
│   ├── realsense_d555.jpg             # Intel D555 相机照片
│   └── Luxonis_OAK_D_Pro_PoE.jpg      # Luxonis OAK-D Pro 相机照片
│
└── ggcnn_grasping_demo/               # ★ 核心项目目录
    ├── LICENSE                        # 许可证
    │
    ├── camera/                        # ① 相机驱动层
    ├── grasp/                         # ② 抓取控制层
    ├── ggcnn/                         # ③ GGCNN 神经网络
    ├── models/                        # ④ 预训练模型权重
    ├── example/                       # ⑤ 各平台运行脚本
    ├── multi_arm/                     # ⑥ 三臂协同系统
    └── simulation/                    # ⑦ 仿真平台接口 (★面向仿真平台负责人)
```

---

## 二、camera/ — 相机驱动层

| 文件 | 说明 |
|------|------|
| `rs_camera.py` | **Intel RealSense D435/D555 相机驱动**。封装 `pyrealsense2`，提供对齐的彩色+深度图像流、内参读取、手动曝光控制 |
| `depthai_camera.py` | **Luxonis OAK-D Pro PoE 相机驱动**。封装 `depthai` SDK，支持以太网接口 |
| `utils.py` | **图像拼接工具**。提供 `get_combined_img()` 将彩色图和热力图左右拼接显示 |
| `d435_test_learn.py` | **D435 相机调试工具**。鼠标悬停读深度值、点击获取3D坐标、按 S 保存图像 |
| `__init__.py` | 空文件，标记为 Python 包 |

---

## 三、grasp/ — 抓取控制层

| 文件 | 说明 |
|------|------|
| `ggcnn_torch.py` | **GGCNN PyTorch 推理引擎 ★核心**。深度图预处理（裁剪/修补/缩放）→ 模型前向推理 → 输出4张热力图（质量图/角度sin/角度cos/宽度图）→ 提取峰值像素 → 相机投影得到抓取点 [x,y,z,angle] |
| `robot_grasp.py` | **xArm5/6/7/850 闭环伺服抓取**。基于 RobotGrasp 类，三线程架构（位置轮询/GGCNN命令/状态检查），使用伺服模式(mode 7)在下行过程中持续调整位姿 |
| `robot_grasp_lite6.py` | **xArm Lite6 闭环伺服抓取**。与上类似但针对 Lite6 + 真空吸盘调整参数（更小的工作范围、更低的Z限位） |
| `helpers/matrix_funcs.py` | **坐标变换数学库**。euler2mat() 位姿转4×4矩阵、convert_pose() 坐标系转换、rpy_to_rot() 欧拉角转旋转矩阵 |
| `helpers/covariance.py` | 随机协方差矩阵生成（未被使用，遗留代码） |
| `helpers/__init__.py` | 空文件 |

---

## 四、ggcnn/ — GGCNN 神经网络

### 4.1 模型定义

| 文件 | 说明 |
|------|------|
| `models/ggcnn.py` | **GGCNN 模型定义（原版 RSS 2018）**。3层卷积 + 3层转置卷积，4个输出头（质量/角度sin/角度cos/宽度），约6.2万参数 |
| `models/ggcnn2.py` | **GGCNN2 模型定义（改进版）**。更深：4层卷积 + 2层空洞卷积 + 双线性上采样，更大的感受野 |
| `models/common.py` | 后处理辅助函数 `post_process_output()` |
| `models/__init__.py` | 模型工厂函数 `get_network(name)`，根据名称返回 GGCNN 或 GGCNN2 |

### 4.2 训练与评估

| 文件 | 说明 |
|------|------|
| `train_ggcnn.py` | **训练脚本**。支持 Cornell/Jacquard 数据集，TensorBoard 可视化 |
| `eval_ggcnn.py` | **评估脚本**。计算 IoU 指标，可视化抓取矩形 |
| `eval_ggcnn_test.py` | 评估脚本的调试版（含更多 print 输出） |
| `requirements.txt` | 训练依赖 (torch, numpy, opencv, matplotlib, ...) |

### 4.3 数据处理

| 文件 | 说明 |
|------|------|
| `utils/data/cornell_data.py` | Cornell 抓取数据集加载器 |
| `utils/data/jacquard_data.py` | Jacquard 抓取数据集加载器 |
| `utils/data/grasp_data.py` | 数据集抽象基类 `GraspDatasetBase` |
| `utils/data/__init__.py` | 数据集工厂 `get_dataset(name)` |
| `utils/dataset_processing/grasp.py` | 抓取矩形表示 `GraspRectangles`、`detect_grasps()` |
| `utils/dataset_processing/evaluation.py` | IoU 计算、`plot_output()` 可视化 |
| `utils/dataset_processing/image.py` | 图像处理辅助 |
| `utils/dataset_processing/generate_cornell_depth.py` | 从 Cornell 正样本生成深度图 |
| `utils/visualisation/gridshow.py` | 多图网格显示 |
| `utils/timeit.py` | 计时上下文管理器 |

---

## 五、models/ — 预训练模型权重

| 文件 | 大小 | 说明 |
|------|------|------|
| `ggcnn_epoch_23_cornell` | 260 KB | GGCNN 模型，Cornell 数据集训练 23 轮 |
| `epoch_50_cornell` | 295 KB | **GGCNN2 模型**，Cornell 数据集训练 50 轮 ★实际使用 |

---

## 六、example/ — 各硬件平台运行脚本

```
example/
├── realsense_d435/                   # Intel D435 相机方案
│   ├── README.md / README_ZH.md      # 平台说明
│   ├── requirements_rs.txt           # 依赖清单
│   ├── run_rs_d435_grasp.py          # xArm5/6 原始闭环抓取
│   ├── run_rs_d435_grasp_lite6.py    # Lite6 原始闭环抓取
│   ├── run_rs_d435_grasp_lite6_test.py      # 测试版 (始终全局最大值)
│   ├── run_rs_d435_grasp_lite6_openloop.py  # 开环抓取 (手动按G键触发)
│   ├── run_rs_d435_grasp_lite6_new.py       # 两段式自动抓取 (GGCNN2)
│   ├── run_rs_d435_grasp_lite6_new_best.py  # ★最佳版本 (多聚类+自动恢复)
│   └── grasp_debug.py                      # 调试工具 (仅诊断，不动机器人)
│
├── realsense_d555/                   # Intel D555 相机方案 (更高精度)
│   ├── README.md / README_ZH.md
│   ├── requirements_rs.txt
│   ├── run_rs_d555_grasp.py
│   └── run_rs_d555_grasp_lite6.py
│
└── luxonis_oak_poe/                  # Luxonis OAK-D Pro PoE 方案 (以太网)
    ├── README.md / README_ZH.md
    ├── requirements_depthai.txt
    ├── run_oak_poe_grasp.py
    └── run_oak_poe_grasp_lite6.py
```

### D435 脚本演变关系

| 脚本 | 控制模式 | 特点 |
|------|---------|------|
| `run_rs_d435_grasp_lite6.py` | 闭环伺服 (mode 7) | 官方版，下降过程持续调整，可能漂移 |
| `run_rs_d435_grasp_lite6_openloop.py` | 开环 (mode 0) | 手动按G触发，单次抓取 |
| `run_rs_d435_grasp_lite6_new.py` | 两段式开环 | 粗定位→正上方重拍→精定位→直下 |
| **`run_rs_d435_grasp_lite6_new_best.py`** ★ | 两段式开环 | 上者增强：多物块候选聚类、错误自动恢复、真空检测 |
| `run_rs_d435_grasp_lite6_test.py` | 测试用 | OPEN_LOOP_HEIGHT=9999 |
| `grasp_debug.py` | 无运动 | 诊断工具，只打印坐标 |

---

## 七、multi_arm/ — 三臂协同系统 (★新开发)

| 文件 | 行数 | 说明 |
|------|------|------|
| `__init__.py` | 31 | 包初始化 + 延迟导入策略 |
| `config.py` | 217 | 数据类配置 (ArmConfig 单臂参数、SystemConfig 全局参数、ArmState 状态枚举、ZoneState 区域枚举) + JSON加载 |
| `config_3arms.json` | 114 | 三臂硬件参数（IP、手眼标定、工作区边界、观察位、释放点...） |
| `dominant_cluster.py` | 74 | **候选抓取点网格聚类**。补全原 _new_best.py 缺失的 dominant_cluster() 函数。按XY量化→找频率最高格子→邻域合并→返回质心 |
| `collision_avoidance.py` | 203 | **碰撞检测工具**。区域边界自动划分(线形布局)、点包含判断、区域重叠检测、点到区域距离、轨迹碰撞预判、臂间欧氏距离 |
| `coordinator.py` | 634 | **★中心协调器**。臂注册/状态同步/区域锁租约(30s自动过期)/续约/安全距离监控(10Hz)/死锁检测与自动打破/HAZARD管理/L1-L3分级故障响应 |
| `arm_controller.py` | 752 | **★单臂控制线程**。完整封装 _new_best.py 抓取管线：硬件初始化→主循环(GGCNN推理+聚类+触发)→两段式抓取→吸取检测→搬运放置。增加安全移动包装(每步检查Coordinator)、协调区锁定期续约、故障分级处理 |
| `visualizer.py` | 329 | **三路组合显示窗口**。独立渲染线程(~30Hz)：各臂相机图+GGCNN热力图+全局状态面板(状态/计数/区域占有/臂间实时距离)。按q退出、按r清除HAZARD、按s打印摘要 |
| `run_3arm_grasp.py` | 176 | **主启动入口**。加载JSON配置→初始化Coordinator→创建3个ArmController→启动所有线程→阻塞等待退出→清理资源+打印统计 |

### 多臂协同架构

```
config_3arms.json  ──→  SystemConfig
                              │
                   MultiArmCoordinator (中心协调器)
                   区域锁 | 安全监控 | 死锁检测
                     ↑          ↑          ↑
               ArmController ArmController ArmController
                 Thread-0     Thread-1     Thread-2
                    │           │           │
               RealSense   RealSense   RealSense   (独立D435)
               GGCNN2      GGCNN2      GGCNN2      (独立模型)
               XArmAPI     XArmAPI     XArmAPI     (独立连接)
                    │           │           │
                    └───────────┼───────────┘
                                │
                     MultiArmVisualizer (Thread-3)
                         组合显示窗口
```

### 碰撞避免策略

工作台划分为三个层级：

- **独占区**：每臂拥有自己的矩形工作区，仅自己能进入操作
- **协调区**：相邻两臂之间的重叠条带，任一臂可进入但需持锁（互斥）
- **安全高度**：350mm，穿越他人区域时末端必须保持此高度以上

多重安全防护：软件区域锁 + 安全距离监控(10Hz) + xArm内置碰撞检测

---

## 八、simulation/ — 仿真平台接口 (★面向仿真平台负责人)

| 文件 | 行数 | 说明 |
|------|------|------|
| `__init__.py` | 36 | 包初始化 + 延迟导入, 导出 SimulationClient / TaskBuilder / SimCamera |
| `simulation_client.py` | 270 | **HTTP 客户端**。封装与仿真服务器的全部通信: POST /task (发送任务序列)、POST /get_camera (获取图像)、GET / (连通性检查) |
| `task_builder.py` | 294 | **任务序列构建器**。将 GGCNN 抓取目标 [X,Y,Z,Yaw] 转换为仿真平台可执行的 10 步 pick-and-place 序列 (move/vacuum) |
| `sim_camera.py` | 250 | **仿真摄像头适配器**。从仿真服务器 HTTP 获取图像, base64 解码为 numpy ndarray, 提供与 RealSenseCamera 完全兼容的接口 |
| `run_simulation.py` | 330 | **主启动入口**。复用相机+GGCNN+聚类模块, 替换 xArm 控制为 HTTP 仿真调用。支持三种模式: 真实相机 / 仿真摄像头 / 无相机测试 |

### 仿真模式架构

```
┌─────────────────────────────────────────────────────────────┐
│                   仿真平台 (需仿真负责人实现)                    │
│                  http://192.168.1.121:8080                   │
│                                                             │
│  需实现的 API:                                                │
│   GET  /              → 返回任意响应 (连通性检查)               │
│   POST /task          ← 接收 {"sequence": [...]}             │
│   POST /get_camera    → 返回 {"rgb_image":...,"depth_image":}│
└─────────────────────────────────────────────────────────────┘
        ↑ POST 任务序列                    ↓ 图像 base64
┌─────────────────────────────────────────────────────────────┐
│           SimulationClient (simulation_client.py)             │
│   超时重试 · 指数退避 · 自定义异常 · Session 复用               │
└─────────────────────────────────────────────────────────────┘
        ↑                                    ↓
┌──────────────┐                   ┌──────────────────┐
│  TaskBuilder │                   │    SimCamera      │
│  目标→序列   │                   │  HTTP→numpy适配   │
└──────────────┘                   └──────────────────┘
        ↑                                    ↓
┌─────────────────────────────────────────────────────────────┐
│      现有模块 (无需修改, 直接复用)                              │
│   camera/rs_camera.py  ·  grasp/ggcnn_torch.py               │
│   grasp/helpers/matrix_funcs.py  ·  multi_arm/dominant_cluster.py │
└─────────────────────────────────────────────────────────────┘
```

### 8.1 仿真平台负责人需要了解的接口

#### 仿真平台需实现的 2 个 API

| 端点 | 方法 | 方向 | 请求体 | 响应体 |
|------|------|------|--------|--------|
| `/task` | POST | 接收 | `{"sequence": [action, ...]}` | `{"status": "ok"}` 或其他 |
| `/get_camera` | POST | 返回 | `{}` | `{"success": true, "rgb_image": "<base64>", "depth_image": "<base64>"}` |

#### 任务序列格式 (本系统 POST 到仿真平台的 JSON)

```json
{
    "sequence": [
        {
            "type": "move",
            "params": {"x": 230.0, "y": -50.0, "z": 300.0, "roll": 180.0, "pitch": 0.0, "yaw": -30.0},
            "wait": 0.5
        },
        {
            "type": "vacuum",
            "params": {"on": true},
            "wait": 0.8
        }
    ]
}
```

- **move**: 末端移动到绝对位姿 (x/y/z mm, roll/pitch/yaw 度)
- **vacuum**: 真空吸盘开/关
- **wait**: 秒, 该动作完成后等待时间 (机械臂到位/吸力建立的模拟)

#### 摄像头图像格式 (仿真平台 POST 给本系统的 JSON)

```json
{
    "success": true,
    "rgb_image": "base64编码的JPEG/PNG字节",
    "depth_image": "base64编码的16-bit PNG深度图(单位mm)"
}
```

解码后: RGB → BGR uint8 (480,640,3), Depth → float32 (480,640) 米

#### 本系统启动命令 (仿真平台负责人测试用)

```bash
# 全仿真模式 (仿真摄像头 + 仿真机械臂 — 不需要任何硬件)
python run_simulation.py --sim-camera

# 指定服务器地址
python run_simulation.py --sim-camera --server http://192.168.1.121:8080

# 仅测试通信链路 (无摄像头)
python run_simulation.py --no-camera
```

### 8.2 本系统向仿真平台发送的完整序列 (10步)

| 步 | type | 目标 | wait |
|----|------|------|------|
| 1 | move | 移到目标正上方 (above_z=300mm, yaw=抓取角) | 0.5s |
| 2 | move | 上方静置消抖 | 0.5s |
| 3 | move | 直线下降至抓取高度 (Z≥70mm 安全限位) | 1.0s |
| 4 | vacuum | 开启真空吸取 | 0.8s |
| 5 | move | 抬升至观察高度 (380mm) | 0.5s |
| 6 | move | 平移至释放点上方 yaw=0 | 0.5s |
| 7 | move | 下降至释放高度 | 1.0s |
| 8 | vacuum | 关闭真空释放 | 0.5s |
| 9 | move | 抬升至观察高度 | 0.5s |
| 10 | move | 回到观察位 [200,0,380] | 0.5s |

### 8.3 坐标系统说明 (仿真平台须与此对齐)

- **基坐标系原点**: 机械臂底座中心
- **X 轴**: 前向 (远离机器人)
- **Y 轴**: 左向
- **Z 轴**: 上向
- **姿态**: Roll=180° 表示末端朝下 (ZYX 欧拉角)
- **单位**: 位置 mm, 姿态 度

本系统输出的抓取目标 `[X, Y, Z, 180, 0, Yaw]` 已经是基坐标系下的绝对位姿,
仿真平台可直接用于驱动虚拟机械臂。

### 8.4 关键配置参数 (在 run_simulation.py 顶部)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SIM_SERVER_URL` | `http://192.168.1.121:8080` | 仿真服务器地址 |
| `DETECT_XYZ` | `[200, 0, 380]` | 观察位 (相机拍照位置) mm |
| `RELEASE_XYZ` | `[225, -89, 83]` | 释放/放置点 mm |
| `ABOVE_Z` | `300` | 安全悬停高度 mm |
| `GRASPING_MIN_Z` | `70` | 最低下降限位 mm |
| `GRASPING_RANGE` | `[180, 350, -200, 200]` | 安全抓取区 [x_min,x_max,y_min,y_max] mm |

---

## 九、数据流全景图

```
物理层              推理层                控制层               协调层              显示层
───────            ───────              ───────             ───────            ───────

RealSense D435 ──→ process_depth   ──→ euler2mat       ──→ Coordinator  ──→ Visualizer
或 SimCamera        │                 convert_pose        │  │  │            │
    │               ▼                 cam_to_base         │  │  │            ▼
    │            深度预处理            ArmController ──────┘  │  │       OpenCV 窗口
    │            GGCNN2 推理           Thread×3              │  │      (3路相机拼合)
    │               │                 或 SimulationClient    │  │
    │               ▼                      │                │  │
    │        4张输出热力图          ┌──────┴──────┐    ┌─────┴──┴─────┐
    │        (质量/角度/宽度)       │  区域锁     │    │  安全监控     │
    │               │             │ (租约制)    │    │  (10Hz)      │
    │               ▼             └─────────────┘    └──────────────┘
    │        抓取目标 [X,Y,Z,Yaw]
    │               │
    │    ┌──────────┴──────────┐
    │    ▼                     ▼
    │  xArm Lite6 ×3     SimulationClient
    │  真空吸盘抓取        HTTP POST /task
    │  TI0负压检测         仿真平台执行
```

核心链路（真实模式）：**深度图 → GGCNN2 → 抓取点(x,y,z,angle) → 坐标转换 → 基系目标 → 机械臂执行**

核心链路（仿真模式）：**深度图 → GGCNN2 → 抓取点(x,y,z,angle) → 坐标转换 → 基系目标 → TaskBuilder → HTTP POST → 仿真平台执行**

---

## 十、依赖环境

| 组件 | 版本 | 真实模式 | 仿真模式 |
|------|------|----------|----------|
| Python | ≥3.10 | ✓ 必需 | ✓ 必需 |
| PyTorch | 2.4.1 | ✓ 必需 | ✓ 必需 |
| OpenCV | 4.10.0 | ✓ 必需 | ✓ 必需 |
| NumPy | 1.24.4 | ✓ 必需 | ✓ 必需 |
| scikit-image | 0.21.0 | ✓ 必需 | ✓ 必需 |
| requests | ≥2.25 | — | ✓ 必需 (HTTP 通信) |
| pyrealsense2 | 2.56.5 | ✓ 必需 | ○ 可选 (真实相机模式) |
| xarm-python-sdk | 1.14.7 | ✓ 必需 | — 不需要 |

---

## 十一、快速开始

### 单臂抓取 (真实机械臂)
```bash
cd ggcnn_grasping_demo/example/realsense_d435
# 编辑脚本中的 ROBOT_IP 和手眼标定参数
python run_rs_d435_grasp_lite6_new_best.py
```

### 三臂协同 (真实机械臂)
```bash
cd ggcnn_grasping_demo/multi_arm
# 1. 编辑 config_3arms.json 填入每台机械臂的真实IP和标定参数
# 2. 启动
python run_3arm_grasp.py
```

### 仿真模式 (★仿真平台负责人)

```bash
cd ggcnn_grasping_demo/simulation

# Step 1: 测试通信链路 (不需要任何硬件)
python run_simulation.py --no-camera

# Step 2: 全仿真模式 (仿真摄像头 + 仿真机械臂)
python run_simulation.py --sim-camera

# Step 3: 真实相机 + 仿真机械臂 (需要 RealSense D435)
python run_simulation.py
```

仿真模式只需 `requests` 库, 不需要 `pyrealsense2` 或 `xarm-python-sdk`。

---

## 十二、仿真平台负责人阅读路线

如果你是仿真平台开发者, 按以下优先级阅读:

| 优先级 | 文件 | 需要了解什么 |
|--------|------|-------------|
| ★★★ | `项目输入输出格式说明.md` §9 | 仿真接口的全部输入输出格式 |
| ★★★ | 本文档 §8 (simulation/) | 仿真模块结构和 API 端点 |
| ★★☆ | `simulation/simulation_client.py` | HTTP 请求格式和异常处理 |
| ★★☆ | `simulation/task_builder.py` | 完整 10 步序列生成逻辑 |
| ★☆☆ | `simulation/run_simulation.py` | 主循环流程和配置参数 |
| ★☆☆ | `simulation/sim_camera.py` | 摄像头图像编解码 |
| ★☆☆ | `项目输入输出格式说明.md` §1~§8 | 真实模式上下文 (了解 GGCNN 模型输入输出) |

**仿真平台需要实现的接口:**

```
GET  /              → 返回任意响应 (连通性检查)
POST /task          ← 接收 {"sequence": [move/vacuum动作...]}
POST /get_camera    → 返回 {"success": true, "rgb_image": "<base64>", "depth_image": "<base64>"}
```

**本系统启动入口:** `ggcnn_grasping_demo/simulation/run_simulation.py`
**默认服务器地址:** `http://192.168.1.121:8080` (可通过 `--server` 参数修改)
