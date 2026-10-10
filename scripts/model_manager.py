#!/usr/bin/env python3
"""Check and download gesture detection .nb files declared by models_manifest.yaml."""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

import yaml


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    required = data.get("required_models") or []
    if not isinstance(required, list) or not required:
        raise RuntimeError("manifest required_models must be a non-empty list")
    return data


def model_dir(manifest, override):
    path = override or os.environ.get("TOPSBOT_MODEL_DIR") or manifest.get("install_dir")
    if not path:
        raise RuntimeError("model install directory is not configured")
    return os.path.abspath(os.path.expanduser(path))


def missing_models(manifest, destination):
    return [
        name for name in manifest["required_models"]
        if not os.path.isfile(os.path.join(destination, name))
    ]


def api_post(url, payload, headers=None, debug=False):
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "topsbot-gesture-model-manager/1.0",
    }
    request_headers.update(headers or {})
    if debug:
        print(f"[DEBUG] POST {url}", file=sys.stderr)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def aliyun_download_url(share_url, debug=False):
    parsed = urllib.parse.urlparse(share_url)
    match = re.search(r"/s/([A-Za-z0-9]+)(?:/|$)", parsed.path)
    if not match:
        raise RuntimeError("unable to parse share_id from models_zip_url")
    share_id = match.group(1)
    query = urllib.parse.parse_qs(parsed.query)
    if not parsed.hostname:
        raise RuntimeError("models_zip_url does not contain a hostname")
    domain = (query.get("domainId") or [parsed.hostname.split(".", 1)[0]])[0]
    api_base = f"https://{domain}.api.aliyunfile.com"

    token_response = api_post(
        f"{api_base}/v2/share_link/get_share_token",
        {"share_id": share_id, "expire_sec": 7200},
        debug=debug,
    )
    token = token_response.get("share_token")
    if not token:
        raise RuntimeError("Aliyun API did not return a share token")
    headers = {"x-share-token": token}

    list_response = api_post(
        f"{api_base}/v2/file/list",
        {"share_id": share_id, "parent_file_id": "root", "limit": 100},
        headers,
        debug,
    )
    items = list_response.get("items") or []
    archives = [item for item in items if str(item.get("name", "")).lower().endswith(".zip")]
    item = archives[0] if archives else (items[0] if items else None)
    if not item or not item.get("file_id"):
        raise RuntimeError("Aliyun share does not contain a downloadable archive")
    if debug:
        print(f"[DEBUG] archive={item.get('name', '<unnamed>')}", file=sys.stderr)

    download_response = api_post(
        f"{api_base}/v2/file/get_download_url",
        {"share_id": share_id, "file_id": item["file_id"], "expire_sec": 7200},
        headers,
        debug,
    )
    url = download_response.get("url")
    if not url:
        raise RuntimeError("Aliyun API did not return a download URL")
    return url


def download_models(args):
    manifest = load_manifest(args.manifest)
    destination = model_dir(manifest, args.install_dir)
    share_url = args.url or manifest.get("models_zip_url")
    if not share_url:
        raise RuntimeError("models_zip_url is not configured")
    os.makedirs(destination, exist_ok=True)
    print(f"[INFO] Downloading gesture detection models to {destination}", file=sys.stderr)

    url = aliyun_download_url(share_url, args.debug)
    archive_path = ""
    try:
        with tempfile.NamedTemporaryFile(prefix="topsbot_gesture_", suffix=".zip", delete=False) as output:
            archive_path = output.name
            request = urllib.request.Request(
                url, headers={"User-Agent": "topsbot-gesture-det-model-manager/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response:
                shutil.copyfileobj(response, output)

        with zipfile.ZipFile(archive_path) as archive:
            # Build member dict with both basename and full path
            members = {}
            for info in archive.infolist():
                members[os.path.basename(info.filename)] = info
                members[info.filename] = info  # Also index by full path

            optional = manifest.get("optional_models") or []
            model_mapping = manifest.get("model_mapping") or {}

            for name in set(manifest["required_models"] + optional):
                # Check if there's a mapping (archive name different from final name)
                archive_name = model_mapping.get(name, name)
                info = members.get(archive_name)
                if info is None:
                    continue
                target = os.path.join(destination, name)
                with archive.open(info) as source, open(target, "wb") as output:
                    shutil.copyfileobj(source, output)
                print(f"[INFO] Ready: {target}", file=sys.stderr)
    finally:
        if archive_path:
            try:
                os.unlink(archive_path)
            except FileNotFoundError:
                pass

    missing = missing_models(manifest, destination)
    if missing:
        raise RuntimeError("archive is missing required models: " + ", ".join(missing))
    return 0


def ensure_models(args):
    manifest = load_manifest(args.manifest)
    destination = model_dir(manifest, args.install_dir)
    missing = missing_models(manifest, destination)
    if not missing:
        print(f"[INFO] All required models present under {destination}")
        return 0
    print(f"[INFO] Missing models under {destination}: {' '.join(missing)}", file=sys.stderr)
    enabled = str(args.auto_download).lower() in ("true", "1", "yes") and not args.no_download
    if not enabled:
        raise RuntimeError("model files are missing and auto download is disabled")
    download_args = argparse.Namespace(
        manifest=args.manifest,
        install_dir=destination,
        url=None,
        debug=args.debug,
    )
    return download_models(download_args)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--manifest", required=True)
    download.add_argument("--install-dir")
    download.add_argument("--url")
    download.add_argument("--debug", action="store_true")
    download.set_defaults(handler=download_models)

    ensure = subparsers.add_parser("ensure")
    ensure.add_argument("--manifest", required=True)
    ensure.add_argument("--models-dir", "--install-dir", dest="install_dir")
    ensure.add_argument("--auto-download", default="true")
    ensure.add_argument("--no-download", action="store_true")
    ensure.add_argument("--debug", action="store_true")
    ensure.set_defaults(handler=ensure_models)
    return parser


def main():
    args = make_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
