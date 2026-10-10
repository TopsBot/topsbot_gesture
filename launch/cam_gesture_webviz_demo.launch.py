# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

"""Full live pipeline: usb_cam -> Gesture Detection -> WebViz overlay."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.utilities import perform_substitutions
from launch_ros.substitutions import FindPackageShare


def _as_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _launch_setup(context, *args, **kwargs):
    usb_params = LaunchConfiguration("usb_cam_params_file").perform(context)
    gesture_params = LaunchConfiguration("gesture_params_file").perform(context)
    webviz_params = LaunchConfiguration("webviz_params_file").perform(context)
    http_port = LaunchConfiguration("http_port").perform(context)
    models_dir = LaunchConfiguration("models_dir").perform(context)
    auto_download = LaunchConfiguration("auto_download_models").perform(context)
    enable_webviz = _as_bool(LaunchConfiguration("enable_webviz").perform(context))
    # Disable webviz's ta-cv hardware JPEG encoder. Boards without
    # /dev/carveout-lookup (e.g. mes20) would otherwise spam
    # "Failed to open /dev/carveout-lookup" at every frame before falling
    # back to OpenCV imencode; passing it here keeps the upstream
    # tb_img_gesture.yaml untouched (we only own the gesture package).
    webviz_enable_tacv = LaunchConfiguration("webviz_enable_tacv_jpeg").perform(context)

    usb_cam_launch = perform_substitutions(context, [
        PathJoinSubstitution([
            FindPackageShare("topsbot_usb_cam"), "launch", "usb_cam.launch.py",
        ]),
    ])
    gesture_launch = perform_substitutions(context, [
        PathJoinSubstitution([
            FindPackageShare("topsbot_gesture_det"), "launch", "gesture_det.launch.py",
        ]),
    ])
    webviz_launch = perform_substitutions(context, [
        PathJoinSubstitution([
            FindPackageShare("topsbot_webviz"), "launch", "webviz.launch.py",
        ]),
    ])

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(usb_cam_launch),
            launch_arguments={"params_file": usb_params}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gesture_launch),
            launch_arguments={
                "params_file": PathJoinSubstitution([
                    FindPackageShare("topsbot_gesture_det"), "config", gesture_params,
                ]),
                "models_dir": models_dir,
                "auto_download_models": auto_download,
            }.items(),
        ),
    ]

    if enable_webviz:
        webviz_launch_args = {
            "params_file": webviz_params,
            "http_port": http_port,
        }
        if webviz_enable_tacv:
            webviz_launch_args["enable_tacv_jpeg"] = webviz_enable_tacv
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(webviz_launch),
                launch_arguments=webviz_launch_args.items(),
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "usb_cam_params_file",
            default_value="mjpeg2nv12_640x480_zc.yaml",
            description="topsbot_usb_cam NV12 zero-copy scene YAML",
        ),
        DeclareLaunchArgument(
            "gesture_params_file",
            default_value="tb_img_gesture_det.yaml",
            description="Gesture detection ai_infer parameter YAML",
        ),
        DeclareLaunchArgument(
            "models_dir",
            default_value="~/topsbot/models",
            description="Directory containing gesture detection .nb models",
        ),
        DeclareLaunchArgument(
            "auto_download_models",
            default_value="true",
            description="Download missing gesture detection models",
        ),
        DeclareLaunchArgument(
            "enable_webviz",
            default_value="true",
            description="Start browser visualization",
        ),
        DeclareLaunchArgument(
            "webviz_params_file",
            default_value="tb_img_gesture.yaml",
            description="topsbot_webviz gesture overlay YAML",
        ),
        DeclareLaunchArgument(
            "http_port",
            default_value="8000",
            description="WebViz static HTTP port",
        ),
        DeclareLaunchArgument(
            "webviz_enable_tacv_jpeg",
            default_value="false",
            description=(
                "Pass-through to topsbot_webviz/launch/webviz.launch.py. "
                "Default 'false' avoids 'Failed to open /dev/carveout-lookup' "
                "spam on boards without ta-cv dmabuf heap; set 'true' if the "
                "target board has /dev/carveout-lookup."
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
