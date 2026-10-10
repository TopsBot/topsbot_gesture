#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${script_dir}/../config/models_manifest.yaml"
exec python3 "${script_dir}/model_manager.py" download --manifest "${manifest}" "$@"
