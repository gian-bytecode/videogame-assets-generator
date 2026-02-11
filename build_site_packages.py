#!/usr/bin/env python3
# %% [markdown]
# # 📦 Videogame Assets Generator — Build & Upload site_packages
# **Run on Google Colab with GPU runtime.**
#
# For each package:
# 1. `pip install --target` into `/content/site_packages`
# 2. Hash local files, diff against VPS to find new/changed files
# 3. Zip only the diff → upload a single archive → server extracts
# 4. Mark the package as completed on the VPS
#
# On re-run (e.g. after Colab timeout), asks the VPS which packages are
# already done and **resumes from where it left off**.

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit before running
# ═══════════════════════════════════════════════════════════════════════════════
# %%
from __future__ import annotations
# VPS server address (must match server config — use wss:// with TLS)
VPS_URL = "wss://YOUR_VPS_IP_OR_DOMAIN:9999"
if "YOUR_VPS_IP_OR_DOMAIN" in VPS_URL:
    VPS_URL=f"wss://{input('Enter VPS IP or domain: ')}:9999"

# Local install target (Colab ephemeral disk — NOT Drive)
TARGET_DIR = "/content/site_packages"

# Upload path prefix on the VPS (relative to file_root in config.yaml)
REMOTE_PREFIX = "site_packages"

# Chunk size for uploads (must match server's max_message_size)
UPLOAD_CHUNK = 1 * 1024 * 1024  # 2 MiB

# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGE LIST — Each entry = one pip install + upload cycle
# ═══════════════════════════════════════════════════════════════════════════════
# %%

# fmt: off
PACKAGES: list[dict] = [
    # ── 0. Torch (forced into target — CUDA compiled libs need exact version) ──
    {"name": "torch",               "pip": "torch",               "extra": "--index-url https://download.pytorch.org/whl/cu121"},
    {"name": "torchvision",         "pip": "torchvision",         "extra": "--index-url https://download.pytorch.org/whl/cu121"},
    {"name": "torchaudio",          "pip": "torchaudio",          "extra": "--index-url https://download.pytorch.org/whl/cu121"},
    # ── 1. Core ML ──
    {"name": "transformers",        "pip": "transformers"},
    {"name": "diffusers",           "pip": "diffusers"},
    {"name": "accelerate",          "pip": "accelerate"},
    {"name": "safetensors",         "pip": "safetensors"},
    {"name": "huggingface_hub",     "pip": "huggingface_hub"},
    # ── 2. TRELLIS deps ──
    {"name": "kaolin",              "pip": "kaolin",              "extra": "-f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html"},
    # nvdiffrast viene installato automaticamente da TRELLIS (richiede build C++/CUDA)
    {"name": "xatlas",              "pip": "xatlas"},
    {"name": "plyfile",             "pip": "plyfile"},
    {"name": "imageio",             "pip": "imageio"},
    # ── 3. TRELLIS ──
    # TRELLIS non è un pacchetto pip — va clonato manualmente:
    #   git clone https://github.com/microsoft/TRELLIS.git
    # ── 4. StableNormal (git) ──
    {"name": "stablenormal",        "pip": "git+https://github.com/Stable-X/StableNormal.git"},
    # ── 5. torch-geometric + extensions ──
    {"name": "torch-geometric",     "pip": "torch-geometric",     "extra": "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"},
    {"name": "torch-scatter",       "pip": "torch-scatter",       "extra": "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"},
    {"name": "torch-sparse",        "pip": "torch-sparse",        "extra": "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"},
    {"name": "torch-cluster",       "pip": "torch-cluster",       "extra": "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"},
    {"name": "torch-spline-conv",   "pip": "torch-spline-conv",   "extra": "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"},
    # ── 6. RigNet deps ──
    {"name": "networkx",            "pip": "networkx"},
    {"name": "rtree",               "pip": "rtree"},
    # ── 7. Fish Speech (TTS) ──
    {"name": "fish-speech",         "pip": "fish-speech"},
    # ── 8. AudioCraft (SFX) ──
    {"name": "audiocraft",          "pip": "audiocraft"},
    # ── 9. Quantization ──
    {"name": "bitsandbytes",        "pip": "bitsandbytes"},
    {"name": "optimum-quanto",      "pip": "optimum-quanto"},
    # ── 10. 3D / Mesh ──
    {"name": "trimesh",             "pip": "trimesh"},
    {"name": "pymeshlab",           "pip": "pymeshlab"},
    {"name": "open3d",              "pip": "open3d"},
    {"name": "opencv-python-headless", "pip": "opencv-python-headless"},
    # ── 11. Utilities ──
    {"name": "numpy",               "pip": "numpy"},
    {"name": "scipy",               "pip": "scipy"},
    {"name": "Pillow",              "pip": "Pillow"},
    {"name": "tqdm",                "pip": "tqdm"},
    {"name": "soundfile",           "pip": "soundfile"},
]
# fmt: on

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD ENGINE — Do NOT edit below
# ═══════════════════════════════════════════════════════════════════════════════
# %%



import getpass
import hashlib
import json
import os
import ssl
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# ── Bootstrap: ensure websockets is available ──
def _ensure_websockets() -> None:
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("📦 Installing websockets …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "websockets"],
            stdout=subprocess.DEVNULL,
        )

_ensure_websockets()
import websockets.sync.client  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_directory(root: Path) -> dict[str, dict[str, Any]]:
    """Return {relative_posix_path: {sha256, size}} for all files under root.

    Hashing is parallelised across CPU cores for speed.
    """
    if not root.is_dir():
        return {}

    # Collect file list first (fast, single-threaded walk)
    file_list: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            full = Path(dirpath) / fname
            rel = full.relative_to(root).as_posix()
            file_list.append((rel, full))

    # Hash in parallel
    result: dict[str, dict[str, Any]] = {}
    workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(sha256_file, full): (rel, full)
            for rel, full in file_list
        }
        for fut in as_completed(futures):
            rel, full = futures[fut]
            try:
                h = fut.result()
                result[rel] = {"sha256": h, "size": full.stat().st_size}
            except OSError:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# VPS Client (upload-capable)
# ─────────────────────────────────────────────────────────────────────────────

class VPSUploadClient:
    """WSS client for the VPS File Server — supports upload & progress."""

    def __init__(self, url: str, password: str):
        self.url = url
        self.password = password
        self.session_id: str | None = None
        self.ws: Any = None

    def connect(self) -> None:
        print(f"  Connecting to {self.url} …")
        ssl_ctx: ssl.SSLContext | None = None
        if self.url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context()
            # Uncomment for self-signed certs:
            # ssl_ctx.check_hostname = False
            # ssl_ctx.verify_mode = ssl.CERT_NONE

        self.ws = websockets.sync.client.connect(
            self.url,
            ssl_context=ssl_ctx,
            max_size=50 * 1024 * 1024,
            open_timeout=30,
            close_timeout=10,
        )
        # Auth
        self.ws.send(json.dumps({"action": "auth", "password": self.password}))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise ConnectionRefusedError(
                f"Auth failed: {resp.get('message', 'unknown')}"
            )
        self.session_id = resp["session_id"]
        print("  🔑 Authenticated.")

    def close(self) -> None:
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    # ── Progress ──

    def get_progress(self) -> list[str]:
        self.ws.send(json.dumps({
            "action": "get_progress",
            "session_id": self.session_id,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise RuntimeError(f"get_progress error: {resp.get('message')}")
        return resp.get("completed_packages", [])

    def mark_package(self, name: str) -> None:
        self.ws.send(json.dumps({
            "action": "mark_package",
            "session_id": self.session_id,
            "package_name": name,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise RuntimeError(f"mark_package error: {resp.get('message')}")

    # ── Upload verify (which files need uploading?) ──

    def upload_verify(self, local_hashes: dict[str, str]) -> list[str]:
        """Send local hashes → get back list of files that need uploading."""
        self.ws.send(json.dumps({
            "action": "upload_verify",
            "session_id": self.session_id,
            "files": local_hashes,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise RuntimeError(f"upload_verify error: {resp.get('message')}")
        return resp.get("to_upload", [])

    # ── Upload a zip archive ──

    def upload_zip(self, zip_path: Path) -> dict[str, Any]:
        """Upload a zip file to VPS for server-side extraction.

        Returns the server response dict (contains 'verified' and 'extracted').
        """
        file_size = zip_path.stat().st_size
        file_hash = sha256_file(zip_path)

        # Begin
        self.ws.send(json.dumps({
            "action": "upload_zip_begin",
            "session_id": self.session_id,
            "sha256": file_hash,
            "size": file_size,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            print(f"    ❌ upload_zip_begin error: {resp.get('message')}")
            return resp

        # Send binary chunks with progress
        sent = 0
        t0 = time.time()
        with open(zip_path, "rb") as f:
            while True:
                chunk = f.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                self.ws.send(chunk)
                sent += len(chunk)
                elapsed = time.time() - t0
                pct = sent / file_size * 100
                speed = sent / (1024**2) / max(elapsed, 0.001)
                print(
                    f"\r    📤 {sent / (1024**2):.1f}/{file_size / (1024**2):.1f} MiB"
                    f" ({pct:.0f}%) — {speed:.1f} MiB/s",
                    end="", flush=True,
                )
        print()  # newline after progress bar

        # Done
        self.ws.send(json.dumps({
            "action": "upload_zip_done",
            "session_id": self.session_id,
            "sha256": file_hash,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            print(f"    ❌ upload_zip_done error: {resp.get('message')}")
        return resp


# ─────────────────────────────────────────────────────────────────────────────
# pip install wrapper
# ─────────────────────────────────────────────────────────────────────────────

def pip_install(pkg: dict, target: str) -> bool:
    """Run pip install for a single package. Returns True on success."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "-q",
        "--target", target,
    ]
    extra = pkg.get("extra", "")
    if extra:
        cmd.extend(extra.split())
    cmd.append(pkg["pip"])

    print(f"    pip install {pkg['pip']} …", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌")
        print(f"    STDERR: {result.stderr[:500]}")
        return False
    print("✅")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Upload site_packages diff to VPS
# ─────────────────────────────────────────────────────────────────────────────

def upload_diff(client: VPSUploadClient, target: Path) -> tuple[int, int]:
    """
    Hash local site_packages, ask server which files differ, zip them,
    send the single archive, and let the server extract.
    Returns (uploaded_count, failed_count).
    """
    print("    🔍 Hashing local files …", end=" ", flush=True)
    local_files = hash_directory(target)
    print(f"{len(local_files)} files")

    # Build {remote_rel: sha256} for verify
    remote_hashes: dict[str, str] = {}
    for rel, info in local_files.items():
        remote_rel = f"{REMOTE_PREFIX}/{rel}"
        remote_hashes[remote_rel] = info["sha256"]

    print("    🔄 Verifying against VPS …", end=" ", flush=True)
    to_upload = client.upload_verify(remote_hashes)
    print(f"{len(to_upload)} files to upload")

    if not to_upload:
        return 0, 0

    # ── Create zip archive (STORED — no compression, speed over size) ──
    print("    📦 Archiving diff …", end=" ", flush=True)
    zip_fd, zip_path_str = tempfile.mkstemp(suffix=".zip")
    os.close(zip_fd)
    zip_path = Path(zip_path_str)

    total_raw = 0
    file_count = 0
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for remote_rel in to_upload:
                local_rel = remote_rel[len(REMOTE_PREFIX) + 1:]  # strip prefix
                local_file = target / local_rel
                if local_file.is_file():
                    zf.write(str(local_file), arcname=remote_rel)
                    total_raw += local_file.stat().st_size
                    file_count += 1

        zip_size = zip_path.stat().st_size
        print(
            f"{file_count} files, "
            f"{total_raw / (1024**2):.1f} MiB"
        )

        # ── Upload the zip ──
        resp = client.upload_zip(zip_path)
        if not resp.get("verified"):
            return 0, file_count

        extracted = resp.get("extracted", 0)
        print(f"    ✅ Server extracted {extracted} files")
        return file_count, 0

    finally:
        zip_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  📦 Build & Upload site_packages                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    password = getpass.getpass("🔑 VPS password: ")
    if not password:
        print("❌ Password cannot be empty.")
        return

    target = Path(TARGET_DIR)
    target.mkdir(parents=True, exist_ok=True)

    # Connect to VPS
    client = VPSUploadClient(VPS_URL, password)
    try:
        client.connect()
    except Exception as exc:
        print(f"❌ Connection failed: {exc}")
        return

    try:
        # Get progress
        completed = client.get_progress()
        total = len(PACKAGES)
        done = len(completed)
        print()
        print(f"  📊 Progress: {done}/{total} packages already completed")
        if completed:
            print(f"     Last: {completed[-1]}")
        print()

        # Determine which packages to process
        remaining = [p for p in PACKAGES if p["name"] not in completed]
        if not remaining:
            print("  ✅ All packages already installed and uploaded!")
            return

        print(f"  📋 Remaining: {len(remaining)} packages")
        print(f"     Starting from: {remaining[0]['name']}")
        print()

        t_total = time.time()

        for idx, pkg in enumerate(remaining, 1):
            pkg_name = pkg["name"]
            pkg_num = done + idx
            print(f"═══ [{pkg_num}/{total}] {pkg_name} ═══")

            # 1. pip install
            t_pkg = time.time()
            ok = pip_install(pkg, TARGET_DIR)
            if not ok:
                print(f"  ❌ pip install failed for {pkg_name} — stopping.")
                print(f"  Re-run this script to retry from {pkg_name}.")
                return

            # 2. Upload diff to VPS
            uploaded, failed = upload_diff(client, target)
            print(f"    📤 Uploaded: {uploaded}, Failed: {failed}")

            if failed > 0:
                print(f"  ⚠️  {failed} files failed to upload for {pkg_name}")
                print(f"  Re-run this script to retry.")
                return

            # 3. Mark complete on server
            client.mark_package(pkg_name)
            elapsed_pkg = time.time() - t_pkg
            print(f"    ✅ {pkg_name} complete ({elapsed_pkg:.0f}s)")
            print()

        elapsed_total = time.time() - t_total
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  ✅ ALL PACKAGES INSTALLED AND UPLOADED                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  Total packages: {total}")
        print(f"  Total time:     {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
        print()

    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        print()
        print("Re-run this script to resume from where it stopped.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
else:
    main()
