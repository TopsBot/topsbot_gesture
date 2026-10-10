# topsbot_gesture_det

[English](README.md)

# 功能介绍

`topsbot_gesture_det` 在 MES20 开发板上做 MediaPipe Hands 的双阶段手势识别。

- **接相机**：实时检测画面中的手掌并识别手势，输出边界框、关键点和分类结果
- **单张图片**：对本地 JPG 图像做端到端手势识别，生成带标注图片
- **浏览器预览**：配合 [`topsbot_webviz`](../topsbot_webviz) 查看相机画面和检测结果

# 物料清单

| 序号 | 名称 | 说明 |
| ---- | ---- | ---- |
| 1 | TOPSBOT MES20 板卡 | NPU（`ta_runtime`）与 dmabuf 环境 |
| 2 | USB 摄像头 | 适配 `topsbot_usb_cam`，建议 MJPEG 640×480 |
| 3 | 手势模型 | `palm_detection_lite_int8.nb`、`hand_landmark_lite_int8.nb` |
| 4 | `topsbot_webviz` | 联合测试时的 Web 预览（可选） |
| 5 | `topsbot_ai_infer` | 插件接口与运行时集成 |

# 编译与安装

## PC 端（交叉编译）

在 `topsbot_cc` 工具链环境中执行，生成供 MES20 使用的 `install/` 目录：

```bash
# PC：topsbot_cc 工具链环境
cd topsbot_cc/topsbot_ws
./tools/build.sh -s 'topsbot_ai_infer topsbot_gesture_det' 
```
将 `install/` 同步到板卡，或直接在板端工作空间中编译（若板端具备完整编译环境）。



| 文件 | 说明 |
| ---- | ---- |
| `lib/libtopsbot_gesture_det_backend.so` | pluginlib 后端，运行时由 `ai_infer_node` 加载 |
| `lib/libtopsbot_gesture_det_utils.so` | letterbox、NMS、手势规则工具库 |
| `lib/topsbot_gesture_det/gesture_det_file_cli` | 单图离线测试 CLI |
| `share/topsbot_gesture_det/gesture_det_plugins.xml` | pluginlib 描述符 |
| `share/topsbot_gesture_det/launch/gesture_det.launch.py` | ROS 2 launch 文件 |

## MES20 板端（安装与加载）

**方式 A：DEB 包（推荐，源可用时）**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**方式 B：加载工作空间**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
```

# 模型文件

模型文件（`.nb`）不随源码分发。首次在板端启动时，会检查 `~/topsbot/models/`，如缺失则根据 `config/models_manifest.yaml` 自动下载 `models.zip` 并解压。

| 模型文件 | 默认路径 |
| ---- | ---- |
| `palm_detection_lite_int8.nb` | `~/topsbot/models/palm_detection_lite_int8.nb` | 
| `hand_landmark_lite_int8.nb` | `~/topsbot/models/hand_landmark_lite_int8.nb` |

离线环境或模型已手动放置时：

```bash
# MES20
ros2 launch topsbot_gesture_det file_cli.launch.py auto_download_models:=false
```

手动下载脚本（维护者需先配置 `config/models_manifest.yaml` 中的 `models_zip_url`）：

```bash
# PC 或 MES20（源码树）
./scripts/download_models.sh
```

# 使用方式

## file 模式（离线单张 JPG）

适用：不接相机，快速验证模型与标注输出。Launch 文件：`file_cli.launch.py`。

**默认启动**

```bash
# MES20
ros2 launch topsbot_gesture_det file_cli.launch.py 
```

| 项 | 默认值 |
|----|--------|
| 模型 | Gesture（`file_gesture_det.yaml`） |
| 输入 | `share/topsbot_gesture_det/imgs/test.jpg` |
| 输出 | `/tmp/topsbot_gesture_det/out_gesture.jpg` |

**一条命令改模型 / 输入 / 输出**（只写需要改的即可）
```bash
# MES20
ros2 launch topsbot_gesture_det file_cli.launch.py 
  params_file:=model.yaml \
  image_file:=/path/to/input.jpg \
  output_image_file:=/path/to/out.jpg
```
| launch 参数 | 作用 |
|-------------|------|
| `params_file` | 默认模型：`file_gesture_det.yaml` |
| `image_file` | 输入 JPG 路径 |
| `output_image_file` | 输出 JPG 路径 |

## 单节点模式（在线推理）

适用：上游已发布图像话题（如 `topsbot_usb_cam`）。Launch：`gesture_det.launch.py`。

**默认启动**（订阅 `/tbmem_img`，发布 `/gesture_detections`）

```bash
# MES20
ros2 launch topsbot_gesture_det gesture_det.launch.py
```
**常用调整**

```bash
# MES20 — 多相机 / 换订阅话题
ros2 launch topsbot_gesture_det gesture_det.launch.py input_topic:=/cam0/tbmem_img

# MES20 — 跳帧降负载
ros2 launch topsbot_gesture_det gesture_det.launch.py frame_skip:=2
```

| launch 参数 | 作用 |
| ----------- | ---- |
| `params_file` | 场景 yaml（模型、`input_msg`、话题、阈值等） |
| `input_topic` | 覆盖订阅话题 |
| `input_msg` | 覆盖输入类型：`tb_img` / `sensor_image` / `file` |
| `frame_skip` | 跳帧，0 表示不跳 |

查看全部 launch 参数：`ros2 launch topsbot_gesture_det gesture_det.launch.py -s`

## 联合测试模式（相机 + gesture + Web 预览）

适用：USB 相机接入、手势检测和浏览器可视化联调。Launch：`cam_gesture_webviz_demo.launch.py`（一次启动 `topsbot_usb_cam`、`topsbot_gesture_det`、`topsbot_webviz`）。

**默认启动**

```bash
# MES20
source /opt/topsbot/setup.bash
ros2 launch topsbot_gesture_det cam_gesture_webviz_demo.launch.py
```

浏览器访问 `http://<板卡IP>:8000/` 即可查看手势检测结果。

**常用调整**

```bash
# MES20 — 仅检测，不进行 Web 预览
ros2 launch topsbot_gesture_det cam_gesture_webviz_demo.launch.py enable_webviz:=false

# MES20 — 自定义 HTTP 端口
ros2 launch topsbot_gesture_det cam_gesture_webviz_demo.launch.py http_port:=8080
```

**确认链路正常**

```bash
# MES20
ros2 topic hz /tbmem_img
ros2 topic hz /gesture_detections
ros2 topic echo /gesture_detections --once
```

启动正常时，日志中会看到类似输出：

```text
[gesture_det_node]: AI worker core=0 healthy=true in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
[gesture_det_node]: AI totals received=.. published=.. rejected=..
```

# 接口说明

## ROS 话题

| 方向 | 话题 | 消息类型 | 说明 |
| ---- | ---- | ---- | ---- |
| 订阅 | `/tbmem_img`（默认） | `tb_img_msgs/TbMsg480P` 等 | `input_msg: tb_img` |
| 订阅 | `/image_raw` | `sensor_msgs/Image` | `input_msg: sensor_image` |
| 发布 | `/gesture_detections` | `tb_det_msgs/TbPerceptionTargets` | 每只手的边界框 + 21 个关键点 |
| 发布 | `/classifications_by_object` | `tb_cls_msgs/TbClassifications` | 每只手的 `handedness` / `gesture` 标签（按配置开关） |

`file` 模式不发布 ROS 话题，结果直接写入输出图片。

## 节点参数

参数配置文件，字段说明如下：

| 分组 | 主要参数 | 说明 |
| ---- | ---- | ---- |
| 输入（在线） | `input_msg`、`input_topic`、`tb_img_profile` | `tb_img` / `sensor_image`；`tb_img_profile` 仅 `tb_img` 时有效 |
| 输入（file） | `image_file`、`output_image_file` | 离线 JPG 路径 |
| 输出 | `publish_detections`、`detections_topic`、`frame_skip` | 检测话题与跳帧 |
| 模型 | `model.path`、`model.class_names[0]` | 阶段一/二模型路径 |
| 后处理 | `postprocess.confidence_threshold`、`postprocess.nms_threshold`、`postprocess.max_detections` | 置信度与去重 |
| 调试 | `enable_profile`、`publish_object_attributes` | 性能输出与手势属性转发 |

场景 yaml 为上述参数的全量实例，一种用法一份文件：

| `params_file` | `input_msg` | 模型 |
|---------------|-------------|------|
| `` | `tb_img` | Mobilenetv2（在线默认） |
| `tb_img_gesture_det.yaml` | `tb_image` | Gesture（在线默认） |
| `file_gesture_det.yaml` | `file` |  Gesture （离线默认） |

## Launch 文件

| Launch | 用途 |
| ------ | ---- |
| `file_cli.launch.py` | 离线 single-image 模式 |
| `gesture_det.launch.py` | 在线单节点推理 |
| `cam_gesture_webviz_demo.launch.py` | 相机 + gesture + Web 预览联测 |

各 launch 参数可通过：`ros2 launch topsbot_gesture_det <launch名> -s` 查看。

---

## 许可

Apache License 2.0 — 见 [LICENSE](./LICENSE)。

欢迎贡献，请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。
