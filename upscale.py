#!/usr/bin/env python3
"""Upscale every image in the current directory with a ComfyUI API workflow.

ComfyUI only writes inside its own output folder. An absolute Save Image
prefix is rewritten to a relative one. Each finished file is then moved
from that output folder into ./upscaled/ in the directory where you
launched the script.

Usage, from the folder that holds the pictures:

    python /path/to/upscale.py "/path/to/SeedVR upscale-0.json"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
UPLOAD_SUBFOLDER = "upscalecomfy"


class ComfyError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every image in the current directory through a ComfyUI workflow."
    )
    parser.add_argument("workflow", help="Path to a ComfyUI API-format workflow JSON")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8188",
        help="ComfyUI base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="upscaled",
        help="Folder for finished images, relative to the current directory (default: %(default)s)",
    )
    parser.add_argument(
        "--comfy-output",
        default=None,
        help="ComfyUI output folder. By default this is taken from the running ComfyUI process.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10800,
        help="Seconds to wait for each image, including time spent queued (default: %(default)s)",
    )
    return parser.parse_args()


def load_prompt(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Workflow not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Workflow is not valid JSON: {path} ({exc})")

    if not isinstance(data, dict):
        raise SystemExit("Workflow JSON must be an object.")
    if isinstance(data.get("nodes"), list):
        raise SystemExit(
            "This file is a ComfyUI canvas workflow. Export it in API format "
            "(the same shape as SeedVR upscale-0.json, with a class_type on each node) "
            "and pass that file."
        )

    def is_prompt(value: object) -> bool:
        return isinstance(value, dict) and bool(value) and all(
            isinstance(node, dict) and "class_type" in node for node in value.values()
        )

    if is_prompt(data):
        return data
    inner = data.get("prompt")
    if is_prompt(inner):
        return inner
    raise SystemExit(
        "Not an API-format workflow. In ComfyUI, export the graph in API format and pass that file."
    )


def find_images(directory: Path) -> list[Path]:
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    ]
    return sorted(images, key=lambda path: path.name.lower())


def is_save_image(class_type: str) -> bool:
    name = class_type.lower().replace(" ", "").replace("_", "")
    if "latent" in name or "model" in name:
        return False
    return name == "saveimage" or name.endswith("saveimage") or ("save" in name and "image" in name)


def node_ids(prompt: dict, predicate) -> list[str]:
    return [
        node_id
        for node_id, node in prompt.items()
        if isinstance(node, dict) and predicate(node.get("class_type", ""))
    ]


def safe_stem(stem: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in "-_." else "_" for ch in stem]
    text = "".join(cleaned).strip("._") or "image"
    return text[:120]


def request(method: str, url: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ComfyError(f"{method} {url} failed ({exc.code}): {format_error_body(body)}") from exc
    except urllib.error.URLError as exc:
        raise ComfyError(f"Cannot reach ComfyUI at {url}: {exc.reason}") from exc
    if not raw:
        return {}
    return json.loads(raw)


def format_error_body(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.strip() or "empty response"
    error = data.get("error") or {}
    lines = [error.get("message") or "Prompt rejected"]
    if error.get("details"):
        lines.append(str(error["details"]))
    for node_id, info in (data.get("node_errors") or {}).items():
        class_type = info.get("class_type", node_id)
        for err in info.get("errors") or []:
            lines.append(
                f"node {node_id} ({class_type}): {err.get('message')}: {err.get('details')}"
            )
    return "\n".join(lines)


def encode_multipart(fields: dict[str, str], filename: str, content: bytes, content_type: str) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode())
        body.extend(b"\r\n")
    safe_name = filename.replace("\\", "\\\\").replace('"', '\\"')
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(content)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return boundary, bytes(body)


def upload_image(server: str, image_path: Path) -> str:
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    boundary, body = encode_multipart(
        {"overwrite": "true", "type": "input", "subfolder": UPLOAD_SUBFOLDER},
        image_path.name,
        image_path.read_bytes(),
        content_type,
    )
    req = urllib.request.Request(
        f"{server}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ComfyError(f"Upload failed for {image_path.name} ({exc.code}): {detail}") from exc
    name = payload.get("name") or image_path.name
    subfolder = payload.get("subfolder") or ""
    if subfolder:
        return f"{subfolder}/{name}"
    return name


def relative_prefix(existing: str, stem: str) -> str:
    """Keep a prefix that stays inside ComfyUI's output folder.

    A leading slash is an absolute path, and ComfyUI rejects saves outside
    its own output directory. Fall back to a relative prefix in that case.
    """
    raw = (existing or "").strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not raw or raw.startswith("/") or raw.startswith("~") or ".." in parts:
        return f"upscaled/{safe_stem(stem)}"
    return raw


def prepare_prompt(template: dict, uploaded_name: str, stem: str) -> tuple[dict, list[str]]:
    prompt = json.loads(json.dumps(template))
    load_ids = node_ids(prompt, lambda class_type: class_type == "LoadImage")
    save_ids = node_ids(prompt, is_save_image)
    if not load_ids:
        raise ComfyError("Workflow has no Load Image node.")
    if not save_ids:
        raise ComfyError("Workflow has no Save Image node.")
    for node_id in load_ids:
        prompt[node_id]["inputs"]["image"] = uploaded_name
    for node_id in save_ids:
        current = prompt[node_id]["inputs"].get("filename_prefix", "")
        prompt[node_id]["inputs"]["filename_prefix"] = relative_prefix(current, stem)
    return prompt, save_ids


def queue_prompt(server: str, prompt: dict) -> str:
    response = request(
        "POST",
        f"{server}/prompt",
        {"prompt": prompt, "client_id": str(uuid.uuid4())},
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise ComfyError(f"ComfyUI did not return a prompt id: {response}")
    return prompt_id


def queue_state(server: str, prompt_id: str) -> str | None:
    payload = request("GET", f"{server}/queue", timeout=30)
    for item in payload.get("queue_running") or []:
        if len(item) > 1 and item[1] == prompt_id:
            return "running"
    for item in payload.get("queue_pending") or []:
        if len(item) > 1 and item[1] == prompt_id:
            return "queued"
    return None


def history_entry(server: str, prompt_id: str) -> dict | None:
    payload = request("GET", f"{server}/history/{prompt_id}", timeout=30)
    entry = payload.get(prompt_id)
    if isinstance(entry, dict):
        return entry
    if "outputs" in payload and "status" in payload:
        return payload
    return None


def format_execution_error(entry: dict) -> str:
    status = entry.get("status") or {}
    lines = []
    for item in status.get("messages") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        event, data = item[0], item[1]
        if event != "execution_error" or not isinstance(data, dict):
            continue
        lines.append(
            f"{data.get('node_type')} #{data.get('node_id')}: {data.get('exception_message')}"
        )
    if lines:
        return "\n".join(lines)
    return str(status.get("status_str") or "execution failed")


def wait_for_prompt(server: str, prompt_id: str, timeout: float) -> dict:
    started = time.monotonic()
    last_report = 0.0
    seen_in_queue = False
    try:
        while True:
            entry = history_entry(server, prompt_id)
            if entry is not None:
                print(flush=True)
                return entry
            state = queue_state(server, prompt_id)
            now = time.monotonic()
            if state is not None:
                seen_in_queue = True
            elif seen_in_queue:
                for _ in range(10):
                    time.sleep(0.5)
                    entry = history_entry(server, prompt_id)
                    if entry is not None:
                        print(flush=True)
                        return entry
                    if queue_state(server, prompt_id) is not None:
                        break
                else:
                    print(flush=True)
                    raise ComfyError("Prompt left the queue without a history entry.")
                continue
            if now - started > timeout:
                raise ComfyError(f"Timed out after {int(timeout)}s waiting for {prompt_id}.")
            if now - last_report >= 10:
                label = state or "waiting"
                print(f"  {label} {int(now - started)}s", flush=True)
                last_report = now
            time.sleep(2)
    except KeyboardInterrupt:
        try:
            request("POST", f"{server}/interrupt", {"prompt_id": prompt_id}, timeout=10)
        except ComfyError:
            pass
        raise


def saved_images(entry: dict, save_ids: list[str]) -> list[dict]:
    outputs = entry.get("outputs") or {}
    found = []
    for node_id in save_ids:
        node_output = outputs.get(node_id) or {}
        for image in node_output.get("images") or []:
            if image.get("filename"):
                found.append(image)
    return found


def comfy_output_path(comfy_output: Path, image: dict) -> Path:
    filename = image.get("filename") or ""
    subfolder = image.get("subfolder") or ""
    if not filename or Path(filename).name != filename or ".." in Path(subfolder).parts:
        raise ComfyError(f"Refusing to move unexpected ComfyUI path: {subfolder}/{filename}")
    root = comfy_output.resolve()
    source = (root / subfolder / filename).resolve()
    if source != root and root not in source.parents:
        raise ComfyError(f"Saved file is outside the ComfyUI output folder: {source}")
    if not source.is_file():
        raise ComfyError(f"ComfyUI reported {source}, but that file is not there.")
    return source


def move_output_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(source, destination)


def local_name(source: Path, image: dict, index: int, count: int) -> str:
    extension = Path(image["filename"]).suffix or ".png"
    if count == 1:
        return f"{source.stem}{extension}"
    return f"{source.stem}_{index:02d}{extension}"


def find_running_comfy_output() -> Path | None:
    """Return the output folder of a running ComfyUI process, if one can be seen."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        args = [part for part in raw.decode("utf-8", "replace").split("\0") if part]
        if not any(part == "main.py" or part.endswith("/main.py") for part in args):
            continue
        try:
            cwd = (entry / "cwd").resolve()
        except OSError:
            continue
        output_dir = None
        if "--output-directory" in args:
            index = args.index("--output-directory")
            if index + 1 < len(args):
                chosen = Path(args[index + 1])
                output_dir = chosen if chosen.is_absolute() else cwd / chosen
        elif (cwd / "main.py").is_file() and (cwd / "nodes.py").is_file():
            output_dir = cwd / "output"
        if output_dir is not None and output_dir.is_dir():
            return output_dir.resolve()
    return None


def check_server(server: str) -> None:
    try:
        request("GET", f"{server}/system_stats", timeout=10)
    except ComfyError as exc:
        raise SystemExit(str(exc))


def main() -> int:
    args = parse_args()
    server = args.server.rstrip("/")
    workflow_path = Path(args.workflow).expanduser().resolve()
    output_dir = (Path.cwd() / args.output).resolve()
    if args.comfy_output:
        comfy_output = Path(args.comfy_output).expanduser().resolve()
    else:
        found = find_running_comfy_output()
        if found is None:
            raise SystemExit(
                "Could not find the running ComfyUI output folder. Pass --comfy-output."
            )
        comfy_output = found
    template = load_prompt(workflow_path)
    if not node_ids(template, lambda class_type: class_type == "LoadImage"):
        raise SystemExit("Workflow has no Load Image node.")
    if not node_ids(template, is_save_image):
        raise SystemExit("Workflow has no Save Image node.")

    images = find_images(Path.cwd())
    if not images:
        raise SystemExit(f"No images found in {Path.cwd()}")

    check_server(server)
    if not comfy_output.is_dir():
        raise SystemExit(f"ComfyUI output folder not found: {comfy_output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Workflow: {workflow_path}")
    print(f"Server:   {server}")
    print(f"Images:   {len(images)} in {Path.cwd()}")
    print(f"Comfy:    {comfy_output}")
    print(f"Output:   {output_dir}")
    print("ComfyUI saves in its output folder. Each result is moved here afterwards.")

    saved_paths: list[Path] = []
    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] {image_path.name}", flush=True)
        uploaded = upload_image(server, image_path)
        print(f"  uploaded {uploaded}", flush=True)
        prompt, save_ids = prepare_prompt(template, uploaded, image_path.stem)
        prefix = prompt[save_ids[0]]["inputs"]["filename_prefix"]
        print(f"  save prefix {prefix}", flush=True)
        prompt_id = queue_prompt(server, prompt)
        print(f"  queued {prompt_id}", flush=True)
        try:
            entry = wait_for_prompt(server, prompt_id, args.timeout)
        except KeyboardInterrupt:
            print("\nInterrupted. The running prompt was cancelled.", file=sys.stderr)
            return 130
        status = (entry.get("status") or {}).get("status_str")
        if status not in (None, "success"):
            print(format_execution_error(entry), file=sys.stderr)
            return 1
        images_out = saved_images(entry, save_ids)
        if not images_out:
            print("Save Image finished without an image.", file=sys.stderr)
            return 1
        for image_index, image in enumerate(images_out, start=1):
            source = comfy_output_path(comfy_output, image)
            destination = output_dir / local_name(image_path, image, image_index, len(images_out))
            move_output_image(source, destination)
            saved_paths.append(destination)
            print(f"  moved {source.name} -> {destination}", flush=True)

    print(f"\nMoved {len(saved_paths)} image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ComfyError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
