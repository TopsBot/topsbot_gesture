# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

DEFAULT_MODELS_DIR = "~/topsbot/models"


def load_params(path):
    """Load ros__parameters from a node params YAML.

    Accepts both a bare mapping and the usual `/**: ros__parameters:` form
    produced by `ros2 param dump`.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("/**", {}).get("ros__parameters", data)


def param_value(params, key, default=None):
    """Read a possibly dot-nested parameter, returning default when absent."""
    if key in params:
        return params[key]
    value = params
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_params_file(pkg_share, name):
    """Resolve a params file name to a path under <pkg_share>/config/.

    Mirrors topsbot_gesture_det's naming: the launch arg that selects the scene
    yaml is `params_file` everywhere (file / single node / joint demo).
    """
    if not name:
        raise RuntimeError("params_file is empty")
    path = name if os.path.isabs(name) else os.path.join(pkg_share, "config", name)
    if not os.path.isfile(path):
        raise RuntimeError(f"gesture_det config file not found: {path}")
    return path


def _gesture_model_paths(params):
    """Return (palm_path, landmark_path) from the gesture params file.

    The palm detector model lives in `model.path`; the hand landmark model
    is carried in `postprocess.class_names` (a one-element list).
    """
    palm = os.path.expanduser(str(param_value(params, "model.path", "") or ""))
    class_names = param_value(params, "postprocess.class_names", [])
    if isinstance(class_names, str):
        class_names = [class_names]
    landmark = os.path.expanduser(str(class_names[0])) if class_names else ""
    return palm, landmark


def ensure_gesture_models(context, params):
    """Validate the configured gesture model files; download when missing.
    """
    backend = str(param_value(params, "model.backend", "") or "").lower()
    if backend != "gesture_det":
        raise RuntimeError(f"expected a gesture_det backend, got: {backend!r}")

    palm_path, landmark_path = _gesture_model_paths(params)
    if not palm_path or not landmark_path:
        raise RuntimeError(
            "gesture params must set model.path and postprocess.class_names")

    missing = [p for p in (palm_path, landmark_path) if not os.path.isfile(p)]
    if not missing:
        return  # Both configured files exist; nothing to do.

    default_dir = os.path.normpath(os.path.expanduser(DEFAULT_MODELS_DIR))
    models_dir = os.path.normpath(os.path.expanduser(
        LaunchConfiguration("models_dir").perform(context) or DEFAULT_MODELS_DIR))

    for path in missing:
        parent = os.path.normpath(os.path.dirname(path))
        if parent not in (default_dir, models_dir):
            message = f"gesture model file not found: {path}"
            if parent == default_dir and models_dir != default_dir:
                message += (
                    " (models_dir was overridden; update model.path/"
                    "postprocess.class_names to the same directory)"
                )
            raise RuntimeError(message)

    auto_download = LaunchConfiguration("auto_download_models").perform(context)
    if auto_download.lower() not in ("true", "1", "yes"):
        raise RuntimeError(
            "gesture models are missing: " + ", ".join(missing)
            + "; run download_models.sh or launch with auto_download_models:=true")

    package_share = get_package_share_directory("topsbot_gesture_det")
    manifest_path = os.path.join(package_share, "config", "models_manifest.yaml")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"models_manifest.yaml not found: {manifest_path}")
    model_manager = os.path.join(package_share, "scripts", "model_manager.py")
    if not os.path.isfile(model_manager):
        raise RuntimeError(f"model_manager.py not found: {model_manager}")

    print(f"[gesture_det] Missing models: {missing}; downloading to {models_dir}",
          file=sys.stderr)
    try:
        subprocess.run([
            sys.executable, model_manager, "download",
            "--manifest", manifest_path,
            "--install-dir", models_dir,
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to download gesture models: {e}") from None

    still_missing = [
        p for p in (palm_path, landmark_path) if not os.path.isfile(p)]
    if still_missing:
        raise RuntimeError(
            "gesture models still missing after download: "
            + ", ".join(still_missing))
    print("[gesture_det] Models ready", file=sys.stderr)