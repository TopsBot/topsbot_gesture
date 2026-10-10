# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

# Offline single-image gesture test (gesture_det_file_cli)
# Merge order: params_file → image_file / output_image_file overrides.

import os
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import (  # noqa: E402
    ensure_gesture_models, load_params, param_value, resolve_params_file)


def _resolve_model_paths(params, context):
    """Return (palm, landmark) paths; explicit legacy overrides win."""
    palm = LaunchConfiguration('palm_model').perform(context)
    palm = palm or str(param_value(params, 'model.path', '') or '')
    landmark = LaunchConfiguration('landmark_model').perform(context)
    if not landmark:
        class_names = param_value(params, 'postprocess.class_names', [])
        if isinstance(class_names, str):
            class_names = [class_names]
        landmark = str(class_names[0]) if class_names else ''
    return os.path.expanduser(palm), os.path.expanduser(landmark)


def _threshold(context, launch_key, params_key, params, fallback):
    """Launch arg override wins; otherwise use the params yaml value."""
    value = LaunchConfiguration(launch_key).perform(context)
    if value:
        return value
    return str(param_value(params, params_key, fallback))


def _launch_setup(context):
    pkg_share = get_package_share_directory('topsbot_gesture_det')
    params_file = resolve_params_file(
        pkg_share, LaunchConfiguration('params_file').perform(context))
    params = load_params(params_file)

    # Ensure the params-file models exist, downloading when missing.
    ensure_gesture_models(context, params)

    palm, landmark = _resolve_model_paths(params, context)
    for label, path in (('palm', palm), ('landmark', landmark)):
        if not path or not os.path.isfile(path):
            raise RuntimeError(f'{label} model file not found: {path}')

    image_file = LaunchConfiguration('image_file').perform(context)
    legacy_input = LaunchConfiguration('input').perform(context)
    input_path = os.path.expanduser(legacy_input or image_file)
    if not input_path or not os.path.exists(input_path):
        raise RuntimeError(
            f'input image not found: {input_path} '
            '(use image_file:= to override)')

    output_dir = (
        LaunchConfiguration('output_image_file').perform(context)
        or LaunchConfiguration('output').perform(context))

    conf = _threshold(context, 'conf', 'postprocess.confidence_threshold', params, 0.5)
    nms = _threshold(context, 'nms', 'postprocess.nms_threshold', params, 0.3)
    lm_conf = _threshold(
        context, 'lm_conf', 'postprocess.landmark_confidence_threshold', params, 0.5)

    executable = os.path.join(
        get_package_prefix('topsbot_gesture_det'),
        'lib', 'topsbot_gesture_det', 'gesture_det_file_cli')

    cmd = [
        executable,
        '--palm', palm,
        '--landmark', landmark,
        '--input', input_path,
        '--conf', conf,
        '--nms', nms,
    ]
    if lm_conf:
        cmd.extend(['--lm-conf', lm_conf])
    if output_dir:
        cmd.extend(['--output', os.path.expanduser(output_dir)])

    return [ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='file_gesture_det.yaml',
            description='Gesture params yaml under config/ (models + thresholds)',
        ),
        DeclareLaunchArgument(
            'image_file',
            default_value='share/topsbot_gesture_det/imgs/test.jpg',
            description='Input image path or directory',
        ),
        DeclareLaunchArgument(
            'output_image_file',
            default_value='/tmp/topsbot_gesture_det/out_gesture.jpg',
            description='Output directory for annotated images (optional)',
        ),
        DeclareLaunchArgument(
            'palm_model',
            default_value='',
            description='[deprecated] Explicit palm model overrides params model.path',
        ),
        DeclareLaunchArgument(
            'landmark_model',
            default_value='',
            description='[deprecated] Explicit landmark model overrides params class_names',
        ),
        DeclareLaunchArgument(
            'input',
            default_value='',
            description='[deprecated] alias for image_file',
        ),
        DeclareLaunchArgument(
            'output',
            default_value='',
            description='[deprecated] alias for output_image_file',
        ),
        DeclareLaunchArgument(
            'conf',
            default_value='',
            description='Confidence threshold (default: params postprocess.confidence_threshold)',
        ),
        DeclareLaunchArgument(
            'nms',
            default_value='',
            description='NMS threshold (default: params postprocess.nms_threshold)',
        ),
        DeclareLaunchArgument(
            'lm_conf',
            default_value='',
            description='Landmark confidence (default: params postprocess.landmark_confidence_threshold)',
        ),
        DeclareLaunchArgument(
            'models_dir',
            default_value='~/topsbot/models',
            description='Directory containing gesture .nb model files',
        ),
        DeclareLaunchArgument(
            'auto_download_models',
            default_value='true',
            description='Download missing models before running',
        ),
        OpaqueFunction(function=_launch_setup),
    ])