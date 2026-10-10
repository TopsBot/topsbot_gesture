# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-09-20)

### Added
- Initial release of gesture detection package
- MediaPipe Hands two-stage gesture recognition (palm detection + hand landmark)
- Support for USB camera real-time detection
- Single image offline detection mode (file CLI)
- Web visualization integration with topsbot_webviz
- Three launch modes: file_cli, gesture_det, cam_gesture_webviz_demo
- Automatic model download and management
- Support for tb_img and sensor_msgs/Image input formats
- Frame skip capability for performance optimization
- Pluginlib-based backend architecture
- 21-point hand keypoint detection
- Gesture classification output
- Configurable confidence and NMS thresholds
