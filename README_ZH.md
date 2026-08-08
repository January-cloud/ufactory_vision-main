# ufactory_vision

## 一、项目解决了什么问题

`ufactory_vision` 是一个**视觉引导的机器人自主抓取系统**，解决以下核心问题：

1. **从图像到抓取动作的端到端自动化** — 给定一张深度图，系统自动推理出最佳抓取点和姿态，控制机械臂完成拾取与放置，无需人工示教或编程。

2. **多臂协同作业的碰撞避免** — 三台机械臂在同一工作台上同时作业时，通过区域锁租约、安全距离监控和死锁检测，确保彼此互不碰撞。

3. **仿真与真机无缝切换** — 同一套视觉推理管线，既可在仿真平台（HTTP 通信）上运行，也可驱动真实 xArm 机械臂，支持零硬件成本的开发调试。

4. **外部指令与视觉感知兼容** — 系统既可根据摄像头图像自主决策抓取目标，也可接收外部系统发来的坐标指令直接执行，两种输入模式共存、自动切换。

**适用场景**：柔性产线中多品种小批量的拾取放置任务、实验室自动化、仓储分拣等。

---

## 二、项目整体逻辑链条

整个系统的数据处理遵循一条清晰的流水线，可概括为四个阶段：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. 感知     │ →  │  2. 推理     │ →  │  3. 决策     │ →  │  4. 执行     │
│  获取RGB-D   │    │  GGCNN2     │    │  聚类+锁定    │    │  机械臂/仿真  │
│  深度图像    │    │  抓取点预测  │    │  坐标变换     │    │  拾取放置    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 阶段 1：感知 — 获取深度图像

相机（RealSense D435/D555、OAK-D PoE、仿真摄像头、本地合成摄像头）采集场景的 RGB 彩色图和深度图。**只有深度图参与推理**，彩色图仅用于可视化。

### 阶段 2：推理 — GGCNN2 神经网络预测抓取点

深度图经过裁剪、空洞修补、缩放后输入 GGCNN2 卷积神经网络。模型输出 4 张热力图：
- **质量图** (pos)：每个像素作为抓取点的评分（0~1）
- **角度图** (cos/sin)：抓取角度的三角函数编码
- **宽度图** (width)：推荐的夹爪张开宽度

对质量图取全局最大值像素，结合角度图解码得到相机坐标系下的抓取位姿 `[x, y, z, angle, width]`。

### 阶段 3：决策 — 多帧聚类 + 坐标变换

单帧推理有噪声。系统维护一个滑动窗口（默认 12 帧），将每帧的抓取候选转换到基坐标系后累积。当缓冲区满时执行网格聚类：找到出现频率最高、空间最集中的簇，质心即为最终抓取目标。这个"稳定锁定"机制有效过滤了传感器噪声和模型抖动。

然后将相机系坐标通过手眼标定矩阵变换为机械臂基坐标系下的绝对位姿 `[X, Y, Z, Roll=180, Pitch=0, Yaw]`。

### 阶段 4：执行 — 两段式拾取放置

**真实模式**：机械臂先平移到目标正上方（阶段 2），相机重新拍照精定位（消除平移误差），然后直线下降吸取，检测负压确认吸住后搬运到释放点。

**仿真模式**：将抓取目标转换为 10 步 `move/vacuum` 动作序列，通过 HTTP POST 发送到仿真服务器执行。

---

## 三、各模块职责

```
ggcnn_grasping_demo/
├── camera/          ① 相机驱动层       — 从物理相机获取 RGB-D 图像
├── grasp/           ② 抓取控制层       — GGCNN 推理引擎 + 坐标变换 + 机械臂伺服
├── ggcnn/           ③ 神经网络层       — GGCNN/GGCNN2 模型定义、训练、评估
├── models/          ④ 预训练权重       — 训练好的 .pth 模型文件
├── example/         ⑤ 各平台运行脚本   — 不同硬件组合的启动入口
├── multi_arm/       ⑥ 多臂协同系统     — 三臂并行抓取 + 碰撞避免
└── simulation/      ⑦ 仿真接口层 — HTTP 仿真通信 + 本地合成摄像头 + 全局摄像头 + 三臂仿真 + 外部坐标输入
```

### ① camera/ — 相机驱动层

| 文件 | 职责 |
|------|------|
| `rs_camera.py` | Intel RealSense D435/D555 驱动。封装 `pyrealsense2`，输出对齐的 RGB-D 图像流和内参矩阵。**仅真实机械臂模式使用** |
| `depthai_camera.py` | Luxonis OAK-D Pro PoE 驱动。封装 `depthai` SDK，支持以太网接口 |
| `utils.py` | 图像拼接工具 `get_combined_img()`：将彩色图和热力图左右拼接为一张 OpenCV 显示图像 |
| `d435_test_learn.py` | D435 调试工具：鼠标悬停读深度值、点击获取 3D 坐标、按 S 保存图像 |

### ② grasp/ — 抓取控制层

| 文件 | 职责 |
|------|------|
| `ggcnn_torch.py` | **★ 推理引擎核心**。深度图预处理（裁剪/修补/缩放）→ 模型前向推理 → 4 通道输出 → 峰值提取 → 相机投影得 `[x,y,z,angle]` |
| `robot_grasp.py` | xArm5/6/7/850 闭环伺服抓取。三线程架构（位置轮询/GGCNN 命令/状态检查），下降过程中持续微调 |
| `robot_grasp_lite6.py` | xArm Lite6 闭环伺服抓取。与上类似但针对 Lite6 + 真空吸盘调整参数 |
| `helpers/matrix_funcs.py` | 坐标变换数学库。`euler2mat()` 欧拉角→4×4 矩阵、`convert_pose()` 坐标系转换 |

### ③ ggcnn/ — 神经网络层

| 文件 | 职责 |
|------|------|
| `models/ggcnn.py` | GGCNN 原版模型（RSS 2018）。3 层卷积 + 3 层转置卷积，约 6.2 万参数 |
| `models/ggcnn2.py` | **GGCNN2 改进版** ★实际使用。4 层卷积 + 2 层空洞卷积 + 双线性上采样，更大感受野 |
| `train_ggcnn.py` | 训练脚本。支持 Cornell/Jacquard 数据集 |
| `eval_ggcnn.py` | 评估脚本。计算 IoU 指标，可视化抓取矩形 |

### ④ models/ — 预训练权重

| 文件 | 说明 |
|------|------|
| `epoch_50_cornell` | **GGCNN2 权重** ★实际使用。Cornell 数据集训练 50 轮，295 KB |

### ⑤ example/ — 各平台运行脚本

每个子目录对应一种硬件组合，包含该平台的依赖文件、README 和启动脚本：

| 目录 | 硬件组合 |
|------|----------|
| `realsense_d435/` | D435 相机 + xArm5/6/Lite6 |
| `realsense_d555/` | D555 相机 + xArm5/6/Lite6 |
| `luxonis_oak_poe/` | OAK-D PoE 相机 + xArm5/6/Lite6 |

### ⑥ multi_arm/ — 多臂协同系统

| 文件 | 职责 |
|------|------|
| `config.py` | 配置数据类（`ArmConfig` 单臂参数、`SystemConfig` 全局参数）和 JSON 加载 |
| `config_3arms.json` | 三臂硬件参数（IP、手眼标定、工作区边界、观察位、释放点） |
| `coordinator.py` | **★ 中心协调器**。臂注册/状态同步/区域锁租约（30s 自动过期）/安全距离监控（10Hz）/死锁检测与打破/HAZARD 管理。使用 `RLock` 可重入锁防止嵌套调用导致的死锁（如 `get_summary()` → `any_hazard()`） |
| `arm_controller.py` | **★ 单臂控制线程**。封装完整抓取管线：硬件初始化→主循环（GGCNN+聚类）→两段式抓取→吸取检测→搬运 |
| `collision_avoidance.py` | 碰撞检测工具。区域边界自动划分、点包含判断、臂间距离、轨迹碰撞预判 |
| `dominant_cluster.py` | 候选抓取点网格聚类。按 XY 量化→找频率最高格子→邻域合并→返回质心 |
| `visualizer.py` | 三路组合显示窗口。独立渲染线程：各臂相机图 + GGCNN 热力图 + 全局状态面板 |
| `run_3arm_grasp.py` | **三臂主启动入口**。加载配置→初始化 Coordinator→创建 3 个 ArmController→启动所有线程 |

**多臂协同架构**：

```
config_3arms.json → SystemConfig
                         │
              MultiArmCoordinator (中心)
              区域锁 | 安全监控 | 死锁检测
                ↑          ↑          ↑
          ArmController ArmController ArmController
            Thread-0     Thread-1     Thread-2
               │           │           │
          RealSense    RealSense    RealSense   (独立 D435)
          GGCNN2       GGCNN2       GGCNN2      (独立模型)
          XArmAPI      XArmAPI      XArmAPI     (独立连接)
               │           │           │
               └───────────┼───────────┘
                           │
                MultiArmVisualizer (Thread-3)
                    组合显示窗口
```

### ⑦ simulation/ — 仿真接口层（★ 新重构）

| 文件 | 职责 |
|------|------|
| `simulation_client.py` | HTTP 客户端。封装与仿真服务器的全部通信：POST /task（发送任务序列，可选 `arm_id` 参数支持多臂）、POST /get_camera（获取图像）、GET /（连通性检查） |
| `task_builder.py` | 任务序列构建器。将 GGCNN 抓取目标 `[X,Y,Z,Yaw]` 转换为仿真平台可执行的 10 步 pick-and-place 序列（move/vacuum） |
| `sim_camera.py` | 仿真摄像头适配器。HTTP 获取图像→base64 解码→numpy ndarray，接口与 RealSenseCamera 完全兼容 |
| `builtin_camera.py` | **★ 本地合成摄像头**。纯 Python 生成模拟 RGB-D 图像（桌面+物体+噪声），零硬件依赖、零服务器依赖，默认摄像头 |
| `global_camera.py` | **★ 全局俯瞰摄像头（新）**。BuiltinCamera 子类，视野覆盖全部三臂工作区（左/中/右三个区域），提供鸟瞰场景总览 |
| `external_input.py` | **★ 外部坐标输入服务器**。后台 HTTP 服务器接收外部抓取坐标（POST /grasp_target），通过线程安全队列传递给主循环 |
| `run_simulation.py` | **★ 单臂仿真主入口**。支持三种摄像头模式 + 外部坐标双输入管道 + GGCNN 推理管线 |
| `config_sim_3arms.json` | **★ 三臂仿真配置（新）**。在 `config_3arms.json` 基础上增加仿真专用字段：`sim_server_url`、每臂 `sim_camera_objects`、`global_camera` 参数 |

**sim_3arm/ — 三臂仿真包（★ 新）**

| 文件 | 职责 |
|------|------|
| `sim_coordinator.py` | **★ 仿真协调器**。薄封装 `MultiArmCoordinator`，增加共享 `SimulationClient`、任务发送串行化（每臂锁 + 全局冷却）、`load_sim_config()` 仿真配置加载 |
| `sim_arm_controller.py` | **★ 单臂仿真控制线程**。每臂独立运行：BuiltinCamera/SimCamera → GGCNN 模型 → TaskBuilder。一段式抓取管线（观察→聚类→锁定→POST /task），功能等价于真机 `ArmController` 但通过 HTTP 通信 |
| `sim_visualizer.py` | **★ 多臂+全局摄像头可视化**。独立渲染线程，2×2 布局：Arm-0 / Arm-1 / Arm-2 / 全局摄像头+状态面板。支持 q=退出 / r=清除HAZARD / s=打印状态 |
| `task_recorder.py` | **★ 生产任务落盘记录器（新）**。线程安全地以 JSON Lines 格式将每次发送的生产任务（识别物体、抓取目标、释放位置、10 步动作序列、发送结果）追加写入磁盘，崩溃不丢数据 |
| `run_sim_3arm.py` | **★ 三臂仿真主启动入口**。加载配置→创建任务落盘记录器→连接仿真服务器→创建全局摄像头→创建 SimCoordinator→创建 3 个 SimArmController→启动可视化→等待退出 |

---

## 四、各部分先后执行顺序

### 4.1 单臂真实模式执行流程

以 `run_rs_d435_grasp_lite6_new_best.py`（最佳版本）为例：

```
步骤 1  硬件初始化
        ├─ 连接 RealSense D435 相机
        ├─ 加载 GGCNN2 预训练模型
        └─ 连接 xArm Lite6 机械臂（使能、清错、归零）

步骤 2  移到观察位
        └─ arm.set_position(DETECT_XYZ) → 相机对准工作区

步骤 3  主循环（每帧 100ms）
        ├─ 3a. camera.get_images() → 获取 RGB + 深度图
        ├─ 3b. ggcnn.get_grasp_img(depth) → GGCNN2 推理
        │      └─ 深度图裁剪(300×300) → 修补NaN → 缩放 → 模型推理
        │         → 4通道热力图 → 全局峰值提取 → 相机系抓取点
        ├─ 3c. cam_result_to_base() → 基坐标系变换
        ├─ 3d. 候选缓冲 + dominant_cluster() 聚类
        │      └─ 12帧滑动窗口 → 网格聚类 → 主簇≥6帧→LOCKED
        └─ 3e. 若 LOCKED 且冷却时间到 → 触发抓取

步骤 4  两段式抓取（触发后执行）
        ├─ 阶段2（精定位）
        │   ├─ 平移至目标正上方 ABOVE_Z 高度
        │   ├─ 静置消抖 1s
        │   └─ 采集 20 帧 → 聚类精定位 → 得到精确目标
        └─ 阶段3（吸取+搬运+放置）
            ├─ 对准 XY + 抓取角度
            ├─ 低速直线下降至抓取 Z
            ├─ 开启真空吸盘
            ├─ 轮询 TI0 负压检测（800ms 超时）
            ├─ 若吸住：
            │   ├─ 抬升至安全高度
            │   ├─ 平移至释放点上方
            │   ├─ 下降至释放高度
            │   ├─ 关闭真空
            │   ├─ 抬升 + 回到观察位
            │   └─ 抓取计数 +1
            └─ 若空抓：关真空 → 抬升 → 重试（最多 3 次）

步骤 5  按 q/ESC 退出
        └─ 关吸盘 → 断机械臂 → 停相机 → 关窗口
```

### 4.2 仿真模式执行流程

```
步骤 1  初始化
        ├─ 连接仿真服务器（检查 / 端点）
        ├─ 初始化摄像头（默认 BuiltinCamera 本地合成）
        ├─ 加载 GGCNN2 模型
        └─ [可选] 启动外部坐标 HTTP 服务器

步骤 2  主循环（每帧 100ms）
        ├─ 2a. 检查外部坐标队列（优先！）
        │      └─ 若有外部坐标 → goal = external_target, stable = True
        │                        → 跳过以下 GGCNN 步骤
        ├─ 2b. camera.get_images() → 获取 RGB + 深度图
        ├─ 2c. ggcnn.get_grasp_img(depth) → GGCNN2 推理
        ├─ 2d. 坐标变换 + 候选缓冲 + 聚类
        └─ 2e. 若稳定锁定 + 冷却时间到 → 触发发送

步骤 3  任务发送（触发后执行）
        ├─ task_builder.build_pick_and_place(goal)
        │   └─ 生成 10 步序列：
        │       1. move 到目标正上方（above_z）
        │       2. move 静置消抖
        │       3. move 直线下降
        │       4. vacuum ON
        │       5. move 抬升至安全高度
        │       6. move 平移到释放点上方
        │       7. move 下降到释放高度
        │       8. vacuum OFF
        │       9. move 抬升
        │      10. move 回到观察位
        └─ simulation_client.post_task(sequence)
            └─ HTTP POST /task → 仿真服务器执行

步骤 4  按 q/ESC 退出
        └─ 停外部输入服务器 → 停摄像头 → 关 HTTP 会话 → 关窗口
```

### 4.3 多臂协同模式执行流程

```
步骤 1  加载配置
        └─ load_config("config_3arms.json") → SystemConfig

步骤 2  启动中心协调器
        └─ MultiArmCoordinator(config)
            ├─ 初始化所有区域（独占区 + 协调区）
            └─ 启动安全监控守护线程（10Hz）

步骤 3  创建 3 个 ArmController（每个在独立线程中）
        ├─ Thread-0 (Arm-Left):  init_hardware → goto_observe → main_loop
        ├─ Thread-1 (Arm-Center): init_hardware → goto_observe → main_loop
        └─ Thread-2 (Arm-Right):  init_hardware → goto_observe → main_loop

        每个 main_loop 内部流程：
        ├─ 获取图像 → GGCNN2 推理 → 聚类 → LOCKED
        ├─ 判断目标在独占区还是协调区
        │   ├─ 独占区 → 直接抓取
        │   └─ 协调区 → request_zone() → 获锁后抓取
        ├─ 两段式抓取（期间定期续约区域锁）
        ├─ 释放协调区 → 回到观察位 → 循环
        └─ 每帧：更新 Coordinator 状态 + 末端位姿

步骤 4  启动可视化线程
        └─ MultiArmVisualizer (Thread-3)
            └─ ~30Hz 渲染：三路相机 + 热力图 + 状态面板

步骤 5  安全监控（Coordinator 后台，持续 10Hz）
        ├─ 臂间距离检查（<100mm → 急停）
        ├─ 位置新鲜度检查（>3s 未更新 → DISCONNECTED）
        ├─ EE 卡死检查（>10s 未移动 → HAZARD）
        └─ 区域租约过期检查（30s 未续约 → 自动回收）

步骤 6  按 q/ESC 退出
        └─ broadcast_stop() → 各臂回观察位 → 清理硬件 → 打印统计
```

### 4.4 三臂仿真模式（★ 新增）

```
步骤 1  加载配置
        ├─ load_sim_config("config_sim_3arms.json") → SimSystemConfig
        └─ 创建 TaskRecorder 生产任务落盘记录器（默认 logs/tasks_<时间>.jsonl）

步骤 2  初始化仿真基础设施
        ├─ SimulationClient（所有臂共享同一 HTTP 会话）
        ├─ GlobalCamera（俯瞰全部三臂工作区的鸟瞰摄像头）
        └─ SimCoordinator（包装 MultiArmCoordinator，复用区域/安全管理逻辑）
           └─ 挂载 TaskRecorder → 任务落盘能力

步骤 3  创建 3 个 SimArmController（每个在独立线程中）
        ├─ Thread-0 (Arm-Left):  BuiltinCamera → GGCNN2 → TaskBuilder → main_loop
        ├─ Thread-1 (Arm-Center): BuiltinCamera → GGCNN2 → TaskBuilder → main_loop
        └─ Thread-2 (Arm-Right):  BuiltinCamera → GGCNN2 → TaskBuilder → main_loop

        每个 main_loop 内部流程：
        ├─ 从本臂摄像头取图 → GGCNN2 推理 → 聚类 → LOCKED
        ├─ 判断目标在独占区还是协调区
        │   ├─ 独占区 → 直接构建任务序列
        │   └─ 协调区 → request_zone() 获锁后构建
        ├─ TaskBuilder.build_pick_and_place() 构建 10 步序列
        ├─ coordinator.send_task(arm_id, seq, meta) → POST /task（串行化 + 全局冷却）
        │   └─ 每次发送后由 TaskRecorder 落盘（JSON Lines：物体类型/抓取目标/
        │      释放位置/10 步动作序列/发送结果，成功与失败均记录）
        ├─ 释放协调区 → 回到循环
        └─ 每帧：更新 Coordinator 状态 + 虚拟末端位姿

步骤 4  启动 SimVisualizer（Thread-4）
        └─ ~30Hz 渲染：2×2 布局
            ┌─────────────┬─────────────┐
            │ Arm-0: 彩色图│ Arm-1: 彩色图│
            │ + 热力图     │ + 热力图     │
            ├─────────────┼─────────────┤
            │ Arm-2: 彩色图│ 全局摄像头    │
            │ + 热力图     │ + 状态面板    │
            └─────────────┴─────────────┘

步骤 5  安全监控（Coordinator 后台，持续 10Hz）
        ├─ 臂间距离检查（<100mm → 急停）
        ├─ 位置新鲜度检查（>3s 未更新 → DISCONNECTED）
        ├─ EE 卡死检查（>10s 未移动 → HAZARD）
        └─ 区域租约过期检查（30s 未续约 → 自动回收）

步骤 6  按 q/ESC 退出
        └─ broadcast_stop() → 控制器停止 → 可视化停止
           → 关闭 TaskRecorder 并打印落盘任务数 → 打印抓取统计
```

### 4.5 数据在各阶段之间的流转格式

```
深度图 (480×640 float32, 米)
    │  阶段2: GGCNN2 推理
    ▼
4通道热力图 (300×300)
    │  峰值提取 + 相机投影
    ▼
相机系抓取点 [x, y, z(米), angle(弧度), width(mm)]
    │  坐标变换 (手眼标定 + 实时位姿)
    ▼
基坐标系目标 [X, Y, Z(mm), Roll=180°, Pitch=0°, Yaw°]
    │
    ├─ 真实模式 → arm.set_position(x, y, z, roll, pitch, yaw)
    │              真空吸取 → 负压检测 → 搬运 → 释放
    │
    ├─ 仿真模式 → TaskBuilder.build_pick_and_place()
    │             10步序列 → HTTP POST /task → 仿真服务器
    │
    ├─ 仿真模式（三臂）→ SimArmController (每臂独立)
    │                  └─ BuiltinCamera → GGCNN2 → 聚类 → LOCKED
    │                  └─ TaskBuilder.build_pick_and_place()
    │                  └─ SimCoordinator.send_task(arm_id, seq)
    │                     └─ POST /task {"arm_id": N, "sequence": [...]}
    │                  └─ GlobalCamera 俯瞰全局场景可视化
    │
    └─ 外部输入 → POST /grasp_target → 直接输入基坐标系目标
                  跳过 GGCNN，与图像路径共用后续执行管线
```

---

## 五、快速开始

### 仿真模式（推荐入门，无需硬件）

```bash
cd ggcnn_grasping_demo/simulation

# 默认启动（本地合成摄像头 + 仿真服务器）
python run_simulation.py --server http://192.168.1.121:8080

# 启用外部坐标输入
python run_simulation.py --server http://192.168.1.121:8080 --ext-input

# 无服务器纯本地测试
python run_simulation.py --no-camera
```

### 三臂仿真（★ 新增，无需硬件）

```bash
cd ggcnn_grasping_demo/simulation/sim_3arm

# 默认启动（本地合成摄像头，每臂独立场景）
python run_sim_3arm.py

# 指定仿真服务器
python run_sim_3arm.py --server http://192.168.1.121:8080

# 每臂从仿真服务器获取图像
python run_sim_3arm.py --sim-camera

# 无相机模式（虚拟深度图，仅测试线程/通信链路）
python run_sim_3arm.py --no-camera

# 自定义生产任务落盘路径（默认 logs/tasks_<时间>.jsonl）
python run_sim_3arm.py --task-log logs/tasks.jsonl

# 禁用生产任务落盘
python run_sim_3arm.py --no-task-log
```

### 单臂真实抓取

```bash
cd ggcnn_grasping_demo/example/realsense_d435
# 编辑脚本中的 ROBOT_IP 和手眼标定参数
python run_rs_d435_grasp_lite6_new_best.py
```

### 三臂协同

```bash
cd ggcnn_grasping_demo/multi_arm
# 编辑 config_3arms.json 填入每台机械臂的真实 IP 和标定参数
python run_3arm_grasp.py
```

---

## 六、硬件要求

| 机械臂型号 | 相机型号 | 末端执行器 |
|-----------|---------|-----------|
| xArm 5/6/7 或 850 | Intel RealSense D435/D555 或 Luxonis OAK-D-Pro-PoE | UFACTORY 机械爪 G1/G2 |
| Lite 6 | Intel RealSense D435 或 Luxonis OAK-D-Pro-PoE | Lite 6 真空吸头 |

**仿真模式无需任何硬件**，仅需 Python ≥ 3.10 + PyTorch + OpenCV。

---

## 七、依赖环境

| 组件 | 版本 | 真实模式 | 仿真模式 |
|------|------|----------|----------|
| Python | ≥ 3.10 | ✓ | ✓ |
| PyTorch | 2.4.1 | ✓ | ✓ |
| OpenCV | 4.10.0 | ✓ | ✓ |
| NumPy | 1.24.4 | ✓ | ✓ |
| scikit-image | 0.21.0 | ✓ | ✓ |
| requests | ≥ 2.25 | — | ✓ |
| pyrealsense2 | 2.56.5 | ✓ | — |
| xarm-python-sdk | 1.14.7 | ✓ | — |

---

## 八、坐标系说明

- **基坐标系原点**：机械臂底座中心
- **X 轴**：前向（远离机器人）
- **Y 轴**：左向
- **Z 轴**：上向
- **姿态**：Roll=180° 表示末端朝下（ZYX 欧拉角）
- **单位**：位置 mm，姿态 度

---

## 九、重要提示

- **TCP/坐标系偏移**：请勿设置 TCP 偏移或坐标系偏移，否则可能导致抓取偏差
- **TCP 负载**：请设置正确的 TCP 负载以避免错误的碰撞检测
- **碰撞检测**：运行前确保已启用碰撞检测，建议灵敏度设为 3 或更高
- **仿真模式**默认使用本地合成摄像头，如需连接仿真服务器获取图像，使用 `--sim-camera`
- **三臂仿真** (`sim_3arm/`) 结构与真机 `multi_arm/` 对齐：每臂独立摄像头 + GGCNN 模型 + 全局俯瞰摄像头，支持与单臂仿真相同的三种摄像头模式 (`--sim-camera` / `--no-camera` / 默认本地合成)
- **生产任务落盘** (`sim_3arm/`)：默认将每次发送的生产任务以 JSON Lines 格式写入 `sim_3arm/logs/tasks_<时间>.jsonl`（含识别物体类型、抓取目标、释放位置、10 步动作序列、发送结果，成功与失败均记录），可用 `--task-log` 指定路径、`--no-task-log` 关闭。每次抓取完成后可复用该文件做产量/良率统计

---

## 十、许可证与致谢

本项目采用 **BSD 3-Clause 许可证**。详情见 [LICENSE](LICENSE) 文件。

基于以下开源项目构建：
- [GGCNN](https://github.com/dougsm/ggcnn) — 抓取卷积神经网络
- [ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping) — Kinova 机械臂抓取参考实现
