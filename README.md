# ufactory_vision

> **揭榜挂帅任务（DG-202607）**：多机械臂协同任务，2026 年暑假揭榜挂帅任务，文件包括机械臂协同算法的主要内容，任务的要求以 PDF 格式保存在主目录下。解决问题的主要思路是将三台机械臂的工作区域进行划分，分成专属区和缓冲区，缓冲区的物块由算法判断后进行任务分配；同时做了某台机械臂故障后任务的重新分配。

[中文版说明 (Chinese Version)](./README_ZH.md)

## 1. What Problem Does This Project Solve

`ufactory_vision` is a **vision-guided autonomous robotic grasping system** that addresses the following core problems:

1. **End-to-end automation from image to grasp action** — Given a single depth image, the system autonomously infers the optimal grasp point and pose, then controls the robot arm to complete pick-and-place operations without manual teaching or programming.

2. **Collision-free multi-arm coordination** — When three robot arms operate simultaneously on the same workbench, the system prevents collisions through zone-lock leases, safety distance monitoring, and deadlock detection.

3. **Seamless simulation-to-real transfer** — The same vision inference pipeline runs both in simulation mode (HTTP-based, zero hardware) and on physical xArm robots, enabling development and debugging without hardware costs.

4. **External commands compatible with visual perception** — The system can autonomously detect grasp targets from camera images, while also accepting direct coordinate commands from external systems. Both input modes coexist and switch automatically.

**Use cases**: Flexible production lines with high-mix low-volume pick-and-place tasks, lab automation, warehouse sorting, etc.

---

## 2. Overall Logic Chain

The data processing follows a clear four-stage pipeline:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Sense    │ →  │  2. Infer    │ →  │  3. Decide    │ →  │  4. Execute  │
│  RGB-Depth   │    │  GGCNN2      │    │  Cluster+Lock │    │  Robot/Sim   │
│  Image       │    │  Grasp Pred. │    │  Coord. Trans │    │  Pick&Place  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Stage 1: Sense — Acquire Depth Image

A camera (RealSense D435/D555, OAK-D PoE, simulation camera, or built-in synthetic camera) captures an RGB color image and a depth image of the scene. **Only the depth image is used for inference**; the color image is for visualization only.

### Stage 2: Infer — GGCNN2 Neural Network Predicts Grasp Points

The depth image is cropped, inpainted for NaN regions, and resized before being fed into the GGCNN2 convolutional neural network. The model outputs 4 heatmaps:
- **Quality map** (pos): A score (0~1) for each pixel as a grasp point
- **Angle maps** (cos/sin): Trigonometric encoding of grasp angle
- **Width map** (width): Recommended gripper opening width

The global maximum pixel in the quality map is extracted, and together with the decoded angle, produces a camera-frame grasp pose `[x, y, z, angle, width]`.

### Stage 3: Decide — Multi-frame Clustering + Coordinate Transform

Single-frame inference is noisy. The system maintains a sliding window (default 12 frames), accumulating grasp candidates converted to the base frame. When the buffer is full, grid clustering is performed: the most frequent and spatially concentrated cluster is found, and its centroid becomes the final grasp target. This "stable lock" mechanism effectively filters sensor noise and model jitter.

The camera-frame coordinates are then transformed via the hand-eye calibration matrix into absolute base-frame pose `[X, Y, Z, Roll=180, Pitch=0, Yaw]`.

### Stage 4: Execute — Two-Stage Pick-and-Place

**Real mode**: The arm first moves above the target (Stage 2), the camera re-photographs for fine localization (eliminating translation error), then the arm descends linearly, activates the vacuum gripper, detects negative pressure to confirm suction, and transports the object to the release point.

**Simulation mode**: The grasp target is converted into a 10-step `move/vacuum` action sequence, sent via HTTP POST to the simulation server for execution.

---

## 3. Module Responsibilities

```
ggcnn_grasping_demo/
├── camera/          ① Camera Driver Layer    — RGB-D acquisition from physical cameras
├── grasp/           ② Grasp Control Layer    — GGCNN inference engine + coordinate transforms + arm servoing
├── ggcnn/           ③ Neural Network Layer   — GGCNN/GGCNN2 model definition, training, evaluation
├── models/          ④ Pretrained Weights     — Trained .pth model files
├── example/         ⑤ Platform Scripts       — Launch scripts for different hardware combinations
├── multi_arm/       ⑥ Multi-Arm System       — 3-arm parallel grasping + collision avoidance
└── simulation/      ⑦ Simulation Interface   — HTTP simulation + local synthetic camera + global camera + 3-arm sim + external coordinate input
```

### ① camera/ — Camera Driver Layer

| File | Responsibility |
|------|---------------|
| `rs_camera.py` | Intel RealSense D435/D555 driver. Wraps `pyrealsense2`, outputs aligned RGB-D streams and intrinsics. **Used only in real robot mode** |
| `depthai_camera.py` | Luxonis OAK-D Pro PoE driver. Wraps `depthai` SDK, supports Ethernet |
| `utils.py` | Image compositing utility `get_combined_img()`: stitches color and heatmap side-by-side for OpenCV display |
| `d435_test_learn.py` | D435 debug tool: mouse hover reads depth, click gets 3D coordinates, press S to save |

### ② grasp/ — Grasp Control Layer

| File | Responsibility |
|------|---------------|
| `ggcnn_torch.py` | **★ Core inference engine**. Depth preprocessing (crop/inpaint/resize) → model forward pass → 4-channel output → peak extraction → camera projection → `[x,y,z,angle]` |
| `robot_grasp.py` | xArm5/6/7/850 closed-loop servo grasping. 3-thread architecture (position polling / GGCNN commands / status checking) |
| `robot_grasp_lite6.py` | xArm Lite6 closed-loop servo grasping. Same as above but tuned for Lite6 + vacuum gripper |
| `helpers/matrix_funcs.py` | Coordinate transform math. `euler2mat()` Euler→4×4 matrix, `convert_pose()` frame conversion |

### ③ ggcnn/ — Neural Network Layer

| File | Responsibility |
|------|---------------|
| `models/ggcnn.py` | Original GGCNN model (RSS 2018). 3 conv + 3 transposed conv layers, ~62K params |
| `models/ggcnn2.py` | **GGCNN2 improved version** ★used in practice. 4 conv + 2 dilated conv + bilinear upsampling, larger receptive field |
| `train_ggcnn.py` | Training script. Supports Cornell/Jacquard datasets |
| `eval_ggcnn.py` | Evaluation script. Computes IoU metrics, visualizes grasp rectangles |

### ④ models/ — Pretrained Weights

| File | Description |
|------|-------------|
| `epoch_50_cornell` | **GGCNN2 weights** ★used in practice. Trained 50 epochs on Cornell dataset, 295 KB |

### ⑤ example/ — Platform Scripts

Each subdirectory corresponds to one hardware combination:

| Directory | Hardware |
|-----------|----------|
| `realsense_d435/` | D435 camera + xArm5/6/Lite6 |
| `realsense_d555/` | D555 camera + xArm5/6/Lite6 |
| `luxonis_oak_poe/` | OAK-D PoE camera + xArm5/6/Lite6 |

### ⑥ multi_arm/ — Multi-Arm Coordination System

| File | Responsibility |
|------|---------------|
| `config.py` | Config dataclasses (`ArmConfig`, `SystemConfig`) and JSON loader |
| `config_3arms.json` | 3-arm hardware parameters (IPs, hand-eye calibrations, workspace zones, observe/release poses) |
| `coordinator.py` | **★ Central coordinator**. Arm registration / state sync / zone-lock leases (30s auto-expire) / safety distance monitoring (10Hz) / deadlock detection / HAZARD management. Uses `RLock` to prevent deadlock from nested method calls (e.g., `get_summary()` → `any_hazard()`). |
| `arm_controller.py` | **★ Per-arm control thread**. Full grasp pipeline: hardware init → main loop (GGCNN + clustering) → two-stage grasp → suction check → transport |
| `collision_avoidance.py` | Collision detection utilities. Zone partitioning, point-in-zone tests, inter-arm distance, trajectory collision prediction |
| `dominant_cluster.py` | Candidate grasp point grid clustering. Quantize XY → find highest-frequency cell → merge neighbors → return centroid |
| `visualizer.py` | 3-way composite display. Independent render thread: per-arm camera views + GGCNN heatmaps + global status panel |
| `run_3arm_grasp.py` | **3-arm main entry point**. Load config → init Coordinator → create 3 ArmControllers → start all threads |

**Multi-arm architecture**:

```
config_3arms.json → SystemConfig
                         │
              MultiArmCoordinator (Central)
              Zone Locks | Safety Monitor | Deadlock Detection
                ↑          ↑          ↑
          ArmController ArmController ArmController
            Thread-0     Thread-1     Thread-2
               │           │           │
          RealSense    RealSense    RealSense   (independent D435)
          GGCNN2       GGCNN2       GGCNN2      (independent models)
          XArmAPI      XArmAPI      XArmAPI     (independent connections)
               │           │           │
               └───────────┼───────────┘
                           │
                MultiArmVisualizer (Thread-3)
                    Composite display window
```

### ⑦ simulation/ — Simulation Interface Layer (★ Refactored)

| File | Responsibility |
|------|---------------|
| `simulation_client.py` | HTTP client. Encapsulates all communication with simulation server: POST /task (send task sequences, with optional `arm_id` for multi-arm), POST /get_camera (fetch images), GET / (health check) |
| `task_builder.py` | Task sequence builder. Converts GGCNN grasp targets `[X,Y,Z,Yaw]` into 10-step pick-and-place sequences (move/vacuum) |
| `sim_camera.py` | Simulation camera adapter. HTTP→base64 decode→numpy ndarray, fully compatible with RealSenseCamera interface |
| `builtin_camera.py` | **★ Local synthetic camera**. Pure Python synthetic RGB-D images (table + objects + noise), zero hardware dependency, zero server dependency, default camera |
| `global_camera.py` | **★ Global overhead camera (new)**. BuiltinCamera variant covering the full 3-arm workspace (left / center / right zones) with bird's-eye view for scene overview |
| `external_input.py` | **★ External coordinate input server**. Background HTTP server receiving grasp coordinates (POST /grasp_target), passes to main loop via thread-safe queue |
| `run_simulation.py` | **★ Single-arm simulation main entry**. Supports 3 camera modes + external coordinate dual-input pipeline + GGCNN inference |
| `config_sim_3arms.json` | **★ 3-arm simulation config (new)**. Extends `config_3arms.json` with simulation-specific fields: `sim_server_url`, per-arm `sim_camera_objects`, `global_camera` parameters |

**sim_3arm/ — 3-Arm Simulation Package (★ New)**

| File | Responsibility |
|------|---------------|
| `sim_coordinator.py` | **★ Simulation coordinator**. Thin wrapper around `MultiArmCoordinator`, adds shared `SimulationClient`, serialized task sending (per-arm lock + global cooldown), and `load_sim_config()` for simulation config files |
| `sim_arm_controller.py` | **★ Per-arm simulation control thread**. Each arm runs its own: BuiltinCamera/SimCamera → GGCNN model → TaskBuilder. Single-stage grasp pipeline (observe → cluster → lock → POST /task), equivalent to real `ArmController` but via HTTP |
| `sim_visualizer.py` | **★ Multi-arm + global camera display**. Independent render thread showing 2×2 layout: Arm-0 / Arm-1 / Arm-2 / global camera + status panel. Supports the same q=Quit / r=ClearHazard / s=Summary keybindings |
| `run_sim_3arm.py` | **★ 3-arm simulation main entry**. Load config → connect to simulation server → create global camera → create SimCoordinator → create 3 × SimArmController → start visualizer → wait for exit |

---

## 4. Execution Order

### 4.1 Single-Arm Real Mode

Using `run_rs_d435_grasp_lite6_new_best.py` (best version):

```
Step 1  Hardware Initialization
        ├─ Connect RealSense D435 camera
        ├─ Load GGCNN2 pretrained model
        └─ Connect xArm Lite6 (enable motion, clear errors, set mode)

Step 2  Move to Observation Pose
        └─ arm.set_position(DETECT_XYZ) → camera aimed at workspace

Step 3  Main Loop (100ms per frame)
        ├─ 3a. camera.get_images() → RGB + depth
        ├─ 3b. ggcnn.get_grasp_img(depth) → GGCNN2 inference
        │      └─ crop depth (300×300) → inpaint NaN → resize → model forward
        │         → 4-channel heatmaps → global peak → camera-frame grasp point
        ├─ 3c. cam_result_to_base() → base-frame transform
        ├─ 3d. Candidate buffer + dominant_cluster() clustering
        │      └─ 12-frame sliding window → grid clustering → main cluster ≥6 frames → LOCKED
        └─ 3e. If LOCKED and cooldown elapsed → trigger grasp

Step 4  Two-Stage Grasp (executed on trigger)
        ├─ Stage 2 (Fine Localization)
        │   ├─ Move above target at ABOVE_Z height
        │   ├─ Settle for 1s
        │   └─ Collect 20 frames → cluster → fine-localized target
        └─ Stage 3 (Suction + Transport + Release)
            ├─ Align XY + grasp angle
            ├─ Slow linear descent to grasp Z
            ├─ Activate vacuum gripper
            ├─ Poll TI0 negative pressure detection (800ms timeout)
            ├─ If suction OK:
            │   ├─ Lift to safe height
            │   ├─ Move above release point
            │   ├─ Descend to release height
            │   ├─ Release vacuum
            │   ├─ Lift + return to observation pose
            │   └─ grasp count +1
            └─ If miss: release vacuum → lift → retry (max 3 times)

Step 5  Press q/ESC to exit
        └─ Release vacuum → disconnect arm → stop camera → close windows
```

### 4.2 Simulation Mode

```
Step 1  Initialization
        ├─ Connect to simulation server (check / endpoint)
        ├─ Initialize camera (default: BuiltinCamera local synthetic)
        ├─ Load GGCNN2 model
        └─ [Optional] Start external coordinate HTTP server

Step 2  Main Loop (100ms per frame)
        ├─ 2a. Check external coordinate queue (PRIORITY!)
        │      └─ If external target available → goal = target, stable = True
        │                                         → skip GGCNN steps below
        ├─ 2b. camera.get_images() → RGB + depth
        ├─ 2c. ggcnn.get_grasp_img(depth) → GGCNN2 inference
        ├─ 2d. Coordinate transform + candidate buffer + clustering
        └─ 2e. If stable lock + cooldown elapsed → trigger send

Step 3  Task Dispatch (executed on trigger)
        ├─ task_builder.build_pick_and_place(goal)
        │   └─ Generate 10-step sequence:
        │       1. move above target (above_z)
        │       2. move settle
        │       3. move linear descent
        │       4. vacuum ON
        │       5. move lift to safe height
        │       6. move above release point
        │       7. move descend to release height
        │       8. vacuum OFF
        │       9. move lift
        │      10. move return to observation pose
        └─ simulation_client.post_task(sequence)
            └─ HTTP POST /task → simulation server executes

Step 4  Press q/ESC to exit
        └─ Stop external input server → stop camera → close HTTP session → close windows
```

### 4.3 Multi-Arm Coordination Mode

```
Step 1  Load Config
        └─ load_config("config_3arms.json") → SystemConfig

Step 2  Start Central Coordinator
        └─ MultiArmCoordinator(config)
            ├─ Initialize all zones (exclusive + coordination)
            └─ Start safety monitor daemon thread (10Hz)

Step 3  Create 3 ArmControllers (each in independent thread)
        ├─ Thread-0 (Arm-Left):  init_hardware → goto_observe → main_loop
        ├─ Thread-1 (Arm-Center): init_hardware → goto_observe → main_loop
        └─ Thread-2 (Arm-Right):  init_hardware → goto_observe → main_loop

        Each main_loop internally:
        ├─ Acquire image → GGCNN2 inference → cluster → LOCKED
        ├─ Determine if target in exclusive or coordination zone
        │   ├─ Exclusive → grasp directly
        │   └─ Coordination → request_zone() → grasp after lock acquired
        ├─ Two-stage grasp (renew zone lease periodically during grasp)
        ├─ Release coordination zone → return to observation → loop
        └─ Every frame: update Coordinator state + end-effector pose

Step 4  Start Visualizer Thread
        └─ MultiArmVisualizer (Thread-3)
            └─ ~30Hz render: 3 camera views + heatmaps + status panel

Step 5  Safety Monitor (Coordinator background, continuous 10Hz)
        ├─ Inter-arm distance check (<100mm → emergency stop)
        ├─ Position freshness check (>3s stale → DISCONNECTED)
        ├─ EE stuck check (>10s not moved → HAZARD)
        └─ Zone lease expiry check (30s unrenewed → auto-reclaim)

Step 6  Press q/ESC to exit
        └─ broadcast_stop() → arms return to observation → cleanup → print stats
```

### 4.4 3-Arm Simulation Mode (★ New)

```
Step 1  Load Config
        └─ load_sim_config("config_sim_3arms.json") → SimSystemConfig

Step 2  Initialize Simulation Infrastructure
        ├─ SimulationClient (shared HTTP session across all arms)
        ├─ GlobalCamera (bird's-eye view covering all 3 workspaces)
        └─ SimCoordinator (wraps MultiArmCoordinator for zone/safety management)

Step 3  Create 3 SimArmControllers (each in independent thread)
        ├─ Thread-0 (Arm-Left):  BuiltinCamera → GGCNN2 → TaskBuilder → main_loop
        ├─ Thread-1 (Arm-Center): BuiltinCamera → GGCNN2 → TaskBuilder → main_loop
        └─ Thread-2 (Arm-Right):  BuiltinCamera → GGCNN2 → TaskBuilder → main_loop

        Each main_loop internally:
        ├─ Get RGB-D from per-arm camera → GGCNN2 inference → cluster → LOCKED
        ├─ Check if target is in exclusive or coordination zone
        │   ├─ Exclusive zone → build task sequence directly
        │   └─ Coordination zone → request_zone() via coordinator → build after lock acquired
        ├─ Build 10-step pick-and-place sequence via TaskBuilder.build_pick_and_place()
        ├─ POST /task via coordinator.send_task(arm_id, sequence) — serialized with global cooldown
        ├─ Release coordination zone → loop
        └─ Every frame: update Coordinator state + virtual end-effector pose

Step 4  Start SimVisualizer (Thread-4)
        └─ ~30Hz render: 2×2 layout
            ┌─────────────┬─────────────┐
            │ Arm-0: color │ Arm-1: color│
            │ + heatmap    │ + heatmap   │
            ├─────────────┼─────────────┤
            │ Arm-2: color │ Global Camera│
            │ + heatmap    │ + Status    │
            └─────────────┴─────────────┘

Step 5  Safety Monitor (Coordinator background, continuous 10Hz)
        ├─ Inter-arm distance check (<100mm → emergency stop)
        ├─ Position freshness check (>3s stale → DISCONNECTED)
        ├─ EE stuck check (>10s not moved → HAZARD)
        └─ Zone lease expiry check (30s unrenewed → auto-reclaim)

Step 6  Press q/ESC to exit
        └─ broadcast_stop() → controllers stop → visualizer stops → print stats
```

### 4.5 Data Flow Across All Stages

```
Depth Image (480×640 float32, meters)
    │  Stage 2: GGCNN2 Inference
    ▼
4-Channel Heatmaps (300×300)
    │  Peak extraction + camera projection
    ▼
Camera-Frame Grasp Point [x, y, z(m), angle(rad), width(mm)]
    │  Coordinate transform (hand-eye calib + live pose)
    ▼
Base-Frame Target [X, Y, Z(mm), Roll=180°, Pitch=0°, Yaw°]
    │
    ├─ Real Mode → arm.set_position(x, y, z, roll, pitch, yaw)
    │              Vacuum suction → pressure check → transport → release
    │
    ├─ Simulation → TaskBuilder.build_pick_and_place()
    │              10-step sequence → HTTP POST /task → simulation server
    │
    ├─ Simulation (3-arm) → SimArmController (per arm)
    │                       └─ BuiltinCamera → GGCNN2 → cluster → LOCKED
    │                       └─ TaskBuilder.build_pick_and_place()
    │                       └─ SimCoordinator.send_task(arm_id, seq)
    │                          └─ POST /task {"arm_id": N, "sequence": [...]}
    │                       └─ GlobalCamera for overall scene visualization
    │
    └─ External Input → POST /grasp_target → direct base-frame target
                        Bypass GGCNN, share downstream execution pipeline
```

---

## 5. Quick Start

### Simulation Mode (recommended entry, no hardware needed)

```bash
cd ggcnn_grasping_demo/simulation

# Default launch (local synthetic camera + simulation server)
python run_simulation.py --server http://192.168.1.121:8080

# Enable external coordinate input
python run_simulation.py --server http://192.168.1.121:8080 --ext-input

# Local test without any server
python run_simulation.py --no-camera
```

### 3-Arm Simulation (★ New, no hardware needed)

```bash
cd ggcnn_grasping_demo/simulation/sim_3arm

# Default launch (local synthetic camera, each arm gets its own)
python run_sim_3arm.py

# Custom simulation server
python run_sim_3arm.py --server http://192.168.1.121:8080

# Use HTTP camera per arm (fetch images from simulation server)
python run_sim_3arm.py --sim-camera

# No-camera mode (virtual depth, test thread/communication links only)
python run_sim_3arm.py --no-camera
```

### Single-Arm Real Grasping

```bash
cd ggcnn_grasping_demo/example/realsense_d435
# Edit ROBOT_IP and hand-eye calibration parameters in the script
python run_rs_d435_grasp_lite6_new_best.py
```

### 3-Arm Coordination

```bash
cd ggcnn_grasping_demo/multi_arm
# Edit config_3arms.json with each arm's real IP and calibration params
python run_3arm_grasp.py
```

---

## 6. Hardware Requirements

| Robot Arm Model | Camera Model | End Effector |
|----------------|-------------|-------------|
| xArm 5/6/7 or 850 | Intel RealSense D435/D555 or Luxonis OAK-D-Pro-PoE | UFACTORY Gripper G1/G2 |
| Lite 6 | Intel RealSense D435 or Luxonis OAK-D-Pro-PoE | Vacuum Gripper Lite |

**Simulation mode requires no hardware**, only Python ≥ 3.10 + PyTorch + OpenCV.

---

## 7. Dependencies

| Component | Version | Real Mode | Simulation |
|-----------|---------|-----------|------------|
| Python | ≥ 3.10 | ✓ | ✓ |
| PyTorch | 2.4.1 | ✓ | ✓ |
| OpenCV | 4.10.0 | ✓ | ✓ |
| NumPy | 1.24.4 | ✓ | ✓ |
| scikit-image | 0.21.0 | ✓ | ✓ |
| requests | ≥ 2.25 | — | ✓ |
| pyrealsense2 | 2.56.5 | ✓ | — |
| xarm-python-sdk | 1.14.7 | ✓ | — |

---

## 8. Coordinate System

- **Base frame origin**: Center of robot arm base
- **X axis**: Forward (away from robot)
- **Y axis**: Left
- **Z axis**: Up
- **Orientation**: Roll=180° means end-effector facing downward (ZYX Euler angles)
- **Units**: Position in mm, orientation in degrees

---

## 9. Important Notes

- **TCP/Coordinate Offset**: Do not set TCP offset or coordinate offset, otherwise code tuning may be needed.
- **TCP Payload**: Set correct TCP payload to avoid false collision detection.
- **Collision Detection**: Ensure collision detection is enabled before running. Recommended sensitivity: 3 or higher.
- **Simulation Mode** uses local synthetic camera by default. Use `--sim-camera` to fetch images from a simulation server.
- **3-Arm Simulation** (`sim_3arm/`) mirrors the real `multi_arm/` structure: each arm has its own camera + GGCNN model + a shared global overhead camera. Supports the same 3 camera modes as single-arm simulation (`--sim-camera` / `--no-camera` / default local synthetic).

---

## 10. License & Acknowledgements

This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE) for details.

Built upon the following open-source projects:
- [GGCNN](https://github.com/dougsm/ggcnn) — Grasping Convolutional Neural Network
- [ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping) — Kinova robot arm grasping reference implementation
