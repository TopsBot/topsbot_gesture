# topsbot_gesture_det

[中文文档](README_cn.md)

# Overview

`topsbot_gesture_det` runs MediaPipe Hands two-stage gesture recognition on the MES20 board.

- **With a camera**: detects hands in the live image and outputs bounding boxes, keypoints, and classification results
- **Single image**: performs end-to-end gesture recognition on a local JPG and saves an annotated image
- **Browser preview**: works with [`topsbot_webviz`](../topsbot_webviz) to view the camera feed and detection overlay


# Bill of Materials

| # | Item | Notes |
| - | ---- | ----- |
| 1 | TOPSBOT MES20 board | NPU (`ta_runtime`) and dmabuf support |
| 2 | USB camera | Compatible with `topsbot_usb_cam`; MJPEG 640×480 recommended |
| 3 | Gesture models | `palm_detection_lite_int8.nb`, `hand_landmark_lite_int8.nb` |
| 4 | `topsbot_webviz` | Web preview for joint testing (optional) |
| 5 | `topsbot_ai_infer` | Plugin interface and runtime integration |

# Build and Install

## PC (cross-compile)

Build in the `topsbot_cc` toolchain environment to generate `install/` for the MES20 board:

```bash
# PC: topsbot_cc toolchain
cd topsbot_cc/topsbot_ws
./tools/build.sh -s 'topsbot_ai_infer topsbot_gesture_det'
```

Sync `install/` to the board, or build directly on the board if a full build environment is available.

## MES20 board (install and source)

**Option A: DEB packages (recommended when available)**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**Option B: workspace overlay**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
```

## Model files

Models (`.nb`) are not shipped with source. On first launch on the board, `~/topsbot/models/` is checked; missing files trigger an automatic download of `models.zip` per `config/models_manifest.yaml`, then extraction.

| Model | Default path |
| ----- | ------------ |
| Palm detection | `~/topsbot/models/palm_detection_lite_int8.nb` |
| Hand landmark | `~/topsbot/models/hand_landmark_lite_int8.nb` |

Offline or pre-installed models:

```bash
# MES20
ros2 launch topsbot_gesture_det file_cli.launch.py \
  image_file:=/path/to/hand.jpg \
  auto_download_models:=false
```

Manual download (maintainers must set `models_zip_url` in `models_manifest.yaml`):

```bash
# PC or MES20 (source tree)
./scripts/download_models.sh
```

# Usage
Scene configs are full yaml files under config/. 

## File mode (offline single JPG)

Use when you need a quick model check without a camera. Launch: `file_cli.launch.py`.

**Default launch**

```bash
# MES20
ros2 launch topsbot_gesture_det file_cli.launch.py
```

| Item | Default |
|------|---------|
| Model | Gesture model (`file_gesture_det.yaml`) |
| Input | `share/topsbot_gesture_det/imgs/test.jpg` |
| Output | `/tmp/topsbot_gesture_det/out_gesture.jpg` |

**One command to change model / input / output** (include only what you need)

```bash
# MES20
ros2 launch topsbot_gesture_det file_cli.launch.py \
  params_file:=file_gesture_det.yaml \
  image_file:=/path/to/input.jpg \
  output_image_file:=/path/to/out.jpg
```

| Launch argument | Purpose |
|-----------------|---------|
| `params_file` | Set the model/config file |
| `image_file` | Input JPG path |
| `output_image_file` | Output JPG path |

## Single-node mode (online inference)

Use when an upstream node already publishes image data, such as `topsbot_usb_cam` or a custom camera node. Launch: `gesture_det.launch.py`.

**Default launch** (subscribes to `/tbmem_img`, publishes `/gesture_detections`)

```bash
# MES20
ros2 launch topsbot_gesture_det gesture_det.launch.py
```

**Common adjustments**

```bash
# MES20 — custom input topic
ros2 launch topsbot_gesture_det gesture_det.launch.py input_topic:=/cam0/tbmem_img

# MES20 — frame skip to reduce load
ros2 launch topsbot_gesture_det gesture_det.launch.py frame_skip:=2
```

| Launch argument | Purpose |
|-----------------|---------|
| `params_file` | Scene yaml (model, `input_msg`, topic, thresholds, …) |
| `input_topic` | Override the subscribed topic |
| `input_msg` | Override input type: `tb_img` / `sensor_image` / `file` |
| `frame_skip` | Frame skip; `0` means no skip |

List all launch arguments: `ros2 launch topsbot_gesture_det gesture_det.launch.py -s`

## Joint test mode (camera + gesture + Web preview)

End-to-end USB camera validation with gesture detection and browser visualization. Launch: `cam_gesture_webviz_demo.launch.py` (starts `topsbot_usb_cam`, `topsbot_gesture_det`, and `topsbot_webviz`).

**Default launch**

```bash
# MES20
source /opt/topsbot/setup.bash
#source <install>/setup.bash
ros2 launch topsbot_gesture_det cam_gesture_webviz_demo.launch.py
```

Open `http://<board-ip>:8000/` in a browser to view the gesture detection results.

Defaults: `usb_cam` `mjpeg2nv12_640x480_zc.yaml` + Gesture `tb_img_gesture_det.yaml` + webviz `tb_img_gesture.yaml`.

**Common adjustments**

```bash
# MES20 — run detection without WebViz
ros2 launch topsbot_gesture_det cam_gesture_webviz_demo.launch.py enable_webviz:=false

# MES20 — custom HTTP port
ros2 launch topsbot_gesture_det cam_gesture_webviz_demo.launch.py http_port:=8080
```

**Verify the pipeline**

```bash
# MES20
ros2 topic hz /tbmem_img
ros2 topic hz /gesture_detections
ros2 topic echo /gesture_detections --once
```

On success, the Gesture node logs look like:

```text
[gesture_det_node]: AI worker core=0 healthy=true in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
[gesture_det_node]: AI totals received=.. published=.. rejected=..
```

# Interface Reference

## ROS topics

| Direction | Topic | Message type | Notes |
|-----------|-------|--------------|-------|
| Subscribe | `/tbmem_img` (default) | `tb_img_msgs/TbMsg480P` etc. | `input_msg: tb_img` |
| Subscribe | `/image_raw` | `sensor_msgs/Image` | `input_msg: sensor_image` |
| Publish | `/gesture_detections` | `tb_det_msgs/TbPerceptionTargets` | Bounding boxes and 21 keypoints for each hand |
| Publish | `/classifications_by_object` | `tb_cls_msgs/TbClassifications` | Handedness / gesture labels as configured |

File mode does not publish ROS topics; it writes the result directly to the output image.

## Node parameters

Grouped summary:

| Group | Main parameters | Notes |
|-------|-----------------|-------|
| Input (online) | `input_msg`, `input_topic`, `tb_img_profile` | `tb_img` / `sensor_image`; `tb_img_profile` only applies when `input_msg` is `tb_img` |
| Input (file) | `image_file`, `output_image_file` | Offline JPG paths |
| Output | `publish_detections`, `detections_topic`, `frame_skip` | Detection topic and frame skipping |
| Model | `model.path`, `model.class_names[0]` | Stage 1 and stage 2 model paths |
| Postprocess | `postprocess.confidence_threshold`, `postprocess.nms_threshold`, `postprocess.max_detections` | Confidence and deduplication |
| Debug | `enable_profile`, `publish_object_attributes` | Timing output and attribute forwarding |

Scene YAML files are full parameter sets for each use case:

| `params_file` | `input_msg` | Model |
|---------------|-------------|-------|
| `tb_img_gesture_det.yaml` | `tb_img` | Gesture model (online default) |
| `file_gesture_det.yaml` | `file` | Gesture model (offline default) |

## Launch files

| Launch | Purpose |
|--------|---------|
| `file_cli.launch.py` | Offline single-image mode |
| `gesture_det.launch.py` | Online single-node inference |
| `cam_gesture_webviz_demo.launch.py` | Camera + Gesture + Web preview |

List launch arguments with: `ros2 launch topsbot_gesture_det <launch-name> -s`

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).

Contributions are welcome; see [CONTRIBUTING.md](./CONTRIBUTING.md).
