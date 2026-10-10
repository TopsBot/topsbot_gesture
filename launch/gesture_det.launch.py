# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import ensure_gesture_models, load_params, resolve_params_file  # noqa: E402


def _resolve_params_file_arg(context):
    """params_file is primary; config_file kept for backward compatibility."""
    legacy = LaunchConfiguration('config_file').perform(context)
    if legacy:
        return legacy
    return LaunchConfiguration('params_file').perform(context)


def _param_overrides(context):
    """single-node overrides; empty arg means use params yaml."""
    overrides = {}
    for key in ('input_topic', 'input_msg', 'tb_img_profile'):
        value = LaunchConfiguration(key).perform(context)
        if value:
            overrides[key] = value
    frame_skip = LaunchConfiguration('frame_skip').perform(context)
    if frame_skip:
        overrides['frame_skip'] = int(frame_skip)
    return overrides


def _launch_setup(context):
    pkg_share = get_package_share_directory('topsbot_gesture_det')
    config_file = resolve_params_file(pkg_share, _resolve_params_file_arg(context))

    params = load_params(config_file)
    # Validate configured model files; download into models_dir when missing.
    ensure_gesture_models(context, params)

    return [
        Node(
            package='topsbot_ai_infer',
            executable='ai_infer_node',
            name='gesture_det_node',
            parameters=[config_file, _param_overrides(context)],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='tb_img_gesture_det.yaml',
            description='Gesture params yaml under config/ (or absolute path)',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='',
            description='Deprecated: alias for params_file',
        ),
        DeclareLaunchArgument(
            'input_topic',
            default_value='',
            description='Override ai_infer input topic (default: params yaml)',
        ),
        DeclareLaunchArgument(
            'input_msg',
            default_value='',
            description='Override ai_infer input message type (default: params yaml)',
        ),
        DeclareLaunchArgument(
            'tb_img_profile',
            default_value='',
            description='Override tb_img profile (default: params yaml)',
        ),
        DeclareLaunchArgument(
            'frame_skip',
            default_value='',
            description='Override frame skip, 0 = off (default: params yaml)',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS2 log level',
        ),
        DeclareLaunchArgument(
            'models_dir',
            default_value='~/topsbot/models',
            description='Directory containing gesture detection .nb models',
        ),
        DeclareLaunchArgument(
            'auto_download_models',
            default_value='true',
            description='Auto download models if missing',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
