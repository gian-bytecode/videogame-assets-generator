#!/usr/bin/env python3
# %% [markdown]
# # 🎮 Videogame Assets Generator — Colab Bootstrap
# **Upload ONLY this file to Google Colab.**
#
# What it does:
# 1. Asks for your VPS password (once)
# 2. Clones / updates the code from GitHub
# 3. Downloads model weights & site-packages from your VPS via WSS
# 4. On re-run: verifies file integrity (SHA-256) and re-downloads only changed files
# 5. When everything is synced, stops — you then run the pipeline notebook/script

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these before running
# ═══════════════════════════════════════════════════════════════════════════════
# %%
from __future__ import annotations
# GitHub repository (HTTPS clone URL)
GITHUB_REPO = "https://github.com/gian-bytecode/videogame-assets-generator.git"
GITHUB_BRANCH = "main"

# VPS server address (use wss:// with TLS, or ws:// for testing only)
VPS_URL = "wss://YOUR_VPS_IP_OR_DOMAIN:9999"
if "YOUR_VPS_IP_OR_DOMAIN" in VPS_URL:
    VPS_URL=f"wss://{input('Enter VPS IP or domain: ')}:9999"


# Local workspace root (on Colab's ephemeral disk)
WORKSPACE = "/content/videogame-assets-generator"

# Subdirectories (must match what the VPS serves)
MODELS_DIR = "models_cache"       # model weights
SITE_PACKAGES_DIR = "site_packages"  # pip --target packages

# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP — Do NOT edit below unless you know what you're doing
# ═══════════════════════════════════════════════════════════════════════════════
# %%



import getpass
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Install minimal deps for bootstrap (websockets only — rest comes from VPS)
# ---------------------------------------------------------------------------
def _ensure_bootstrap_deps() -> None:
    """Install websockets if not already present (only dep for bootstrap)."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("📦 Installing websockets for bootstrap …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "websockets"],
            stdout=subprocess.DEVNULL,
        )

_ensure_bootstrap_deps()
import websockets               # noqa: E402
import websockets.sync.client   # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Hashing utilities
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_directory(root: Path, skip_dirs: set[str] | None = None) -> dict[str, str]:
    """
    Walk a directory and return {relative_posix_path: sha256} for every file.
    Skips hidden dirs/files and any directory names in skip_dirs.
    """
    skip = skip_dirs or set()
    result: dict[str, str] = {}
    if not root.is_dir():
        return result
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune hidden & skipped dirs IN PLACE
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in skip
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            full = Path(dirpath) / fname
            rel = full.relative_to(root).as_posix()
            try:
                result[rel] = sha256_file(full)
            except OSError:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — GitHub sync
# ─────────────────────────────────────────────────────────────────────────────

def github_sync(repo_url: str, branch: str, workspace: Path) -> None:
    """
    Clone or update the repo. On update, uses a temp clone + hash comparison
    to replace only changed files (preserves local-only stuff like weights).
    """
    print()
    print("═" * 60)
    print("  📂 STEP 1 — GitHub code sync")
    print("═" * 60)

    git_dir = workspace / ".git"

    if not git_dir.is_dir():
        # Fresh clone
        print(f"  Cloning {repo_url} → {workspace} …")
        subprocess.run(
            ["git", "clone", "--branch", branch, "--single-branch",
             "--depth", "1", repo_url, str(workspace)],
            check=True,
        )
        print("  ✅ Clone complete.")
        return

    # Repo already exists — incremental update via temp clone + hash diff
    print("  Repository exists. Checking for updates …")

    with tempfile.TemporaryDirectory(prefix="vag_git_") as tmpdir:
        tmp_path = Path(tmpdir) / "repo"
        print(f"  Cloning fresh copy to temp dir …")
        subprocess.run(
            ["git", "clone", "--branch", branch, "--single-branch",
             "--depth", "1", repo_url, str(tmp_path)],
            check=True,
            capture_output=True,
        )

        # Hash all tracked files in both locations (skip gitignored dirs)
        skip = {".git", MODELS_DIR, SITE_PACKAGES_DIR, "__pycache__", "output"}
        print("  Hashing local files …")
        local_hashes = hash_directory(workspace, skip_dirs=skip)
        print("  Hashing remote files …")
        remote_hashes = hash_directory(tmp_path, skip_dirs={".git", "__pycache__"})

        # Find files that need updating
        to_update: list[str] = []
        to_delete: list[str] = []

        for rel, remote_hash in remote_hashes.items():
            local_hash = local_hashes.get(rel)
            if local_hash != remote_hash:
                to_update.append(rel)

        for rel in local_hashes:
            if rel not in remote_hashes:
                # File exists locally but not in repo — could be a deleted tracked file
                # Only delete if it's NOT in a gitignored directory
                parts = Path(rel).parts
                if parts and parts[0] not in skip:
                    to_delete.append(rel)

        if not to_update and not to_delete:
            print("  ✅ Code is up to date — no changes.")
            return

        # Apply updates
        for rel in to_update:
            src = tmp_path / rel
            dst = workspace / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  ↻ Updated: {rel}")

        for rel in to_delete:
            target = workspace / rel
            if target.exists():
                target.unlink()
                print(f"  ✗ Removed: {rel}")

        print(f"  ✅ Synced: {len(to_update)} updated, {len(to_delete)} removed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — VPS connection & authentication
# ─────────────────────────────────────────────────────────────────────────────

class VPSClient:
    """WebSocket client for the VPS File Server."""

    def __init__(self, url: str, password: str):
        self.url = url
        self.password = password
        self.session_id: str | None = None
        self.ws: Any = None

    def connect(self) -> None:
        """Open WSS connection and authenticate."""
        print(f"  Connecting to {self.url} …")

        # SSL context — accept self-signed certs if needed
        ssl_ctx: ssl.SSLContext | None = None
        if self.url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context()
            # For self-signed certs on VPS, uncomment the next line:
            # ssl_ctx.check_hostname = False
            # ssl_ctx.verify_mode = ssl.CERT_NONE

        self.ws = websockets.sync.client.connect(
            self.url,
            ssl_context=ssl_ctx,
            max_size=50 * 1024 * 1024,  # 50 MiB max frame
            open_timeout=30,
            close_timeout=10,
        )

        # Authenticate
        self.ws.send(json.dumps({"action": "auth", "password": self.password}))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise ConnectionRefusedError(
                f"Authentication failed: {resp.get('message', 'unknown error')}"
            )
        self.session_id = resp["session_id"]
        print("  🔑 Authenticated successfully.")

    def close(self) -> None:
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    def get_manifest(self) -> dict[str, dict[str, Any]]:
        """Get the full file manifest from the server."""
        self.ws.send(json.dumps({
            "action": "manifest",
            "session_id": self.session_id,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise RuntimeError(f"Manifest error: {resp.get('message')}")
        return resp["files"]

    def verify_files(self, local_hashes: dict[str, str]) -> list[str]:
        """
        Send local hashes to the server, receive list of files
        that need to be (re-)downloaded.
        """
        self.ws.send(json.dumps({
            "action": "verify",
            "session_id": self.session_id,
            "files": local_hashes,
        }))
        resp = json.loads(self.ws.recv())
        if resp.get("status") != "ok":
            raise RuntimeError(f"Verify error: {resp.get('message')}")
        return resp["to_download"]

    def download_zip(self, paths: list[str], workspace: Path) -> tuple[int, int]:
        """
        Ask the server to build a ZIP_STORED archive of the given paths,
        download it as a single stream, and extract locally.
        Returns (extracted_count, failed_count).
        """
        self.ws.send(json.dumps({
            "action": "download_zip",
            "session_id": self.session_id,
            "paths": paths,
        }))

        # Receive header
        header = json.loads(self.ws.recv())
        if header.get("status") != "ok":
            print(f"    ❌ Server error: {header.get('message')}")
            return 0, len(paths)

        expected_hash = header["sha256"]
        expected_size = header["size"]
        file_count = header.get("file_count", len(paths))

        print(f"  📥 Downloading zip: {file_count} files, {expected_size / (1024**2):.1f} MiB")

        # Stream to temp file with progress
        tmp_path = workspace / ".download_tmp.zip"
        hasher = hashlib.sha256()
        received = 0
        t0 = time.time()

        with open(tmp_path, "wb") as f:
            while received < expected_size:
                data = self.ws.recv()
                if isinstance(data, str):
                    # Could be an error or early completion
                    msg = json.loads(data)
                    if msg.get("status") == "error":
                        print(f"    ❌ Error during transfer: {msg.get('message')}")
                        tmp_path.unlink(missing_ok=True)
                        return 0, len(paths)
                    break
                f.write(data)
                hasher.update(data)
                received += len(data)
                elapsed = time.time() - t0
                speed = received / (1024**2) / max(elapsed, 0.001)
                pct = received / expected_size * 100
                print(
                    f"\r  📥 {received / (1024**2):.1f}/{expected_size / (1024**2):.1f} MiB"
                    f" ({pct:.0f}%) — {speed:.1f} MiB/s",
                    end="", flush=True,
                )
        print()  # newline after progress

        # Receive completion message
        completion = json.loads(self.ws.recv())
        if completion.get("status") != "transfer_complete":
            print(f"    ⚠️  Unexpected completion message")
            tmp_path.unlink(missing_ok=True)
            return 0, len(paths)

        # Verify hash
        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            print(f"    ❌ Zip hash mismatch!")
            tmp_path.unlink(missing_ok=True)
            return 0, len(paths)

        # Extract
        print(f"  📦 Extracting …", end=" ", flush=True)
        extracted = 0
        try:
            import zipfile
            with zipfile.ZipFile(tmp_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    target = workspace / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    extracted += 1
            print(f"{extracted} files")
        except Exception as exc:
            print(f"❌ Extraction error: {exc}")
            return 0, len(paths)
        finally:
            tmp_path.unlink(missing_ok=True)

        return extracted, 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — VPS file sync (weights + site-packages)
# ─────────────────────────────────────────────────────────────────────────────

def vps_sync(vps_url: str, password: str, workspace: Path) -> None:
    """
    Connect to VPS, verify local files against remote manifest,
    download only missing/changed files.
    """
    print()
    print("═" * 60)
    print("  📡 STEP 2 — VPS file sync (weights + packages)")
    print("═" * 60)

    client = VPSClient(vps_url, password)
    try:
        client.connect()

        # Get remote manifest
        print("  📋 Requesting file manifest …")
        manifest = client.get_manifest()
        total_files = len(manifest)
        total_size = sum(f["size"] for f in manifest.values())
        print(f"  📋 Remote: {total_files} files, {total_size / (1024**3):.2f} GiB total")

        # Hash local files that correspond to VPS content
        # VPS serves models_cache/ and site_packages/ — both live under workspace
        print("  🔍 Hashing local files for comparison …")
        local_hashes: dict[str, str] = {}
        for rel_path in manifest:
            local_file = workspace / rel_path
            if local_file.is_file():
                local_hashes[rel_path] = sha256_file(local_file)

        existing = len(local_hashes)
        print(f"  🔍 Found {existing}/{total_files} files locally")

        # Ask server which files differ
        print("  🔄 Verifying integrity with server …")
        to_download = client.verify_files(local_hashes)

        if not to_download:
            print("  ✅ All files are up to date — nothing to download!")
            return

        # Calculate download size
        dl_size = sum(manifest[p]["size"] for p in to_download if p in manifest)
        print(f"  📥 Need to download: {len(to_download)} files ({dl_size / (1024**3):.2f} GiB)")
        print()

        # Download as a single zip archive
        extracted, failed = client.download_zip(to_download, workspace)

        print()
        print(f"  ✅ Download complete: {extracted} extracted, {failed} failed")
        if failed:
            print(f"  ⚠️  {failed} files failed — re-run the bootstrap to retry.")

    finally:
        client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Post-sync setup
# ─────────────────────────────────────────────────────────────────────────────

def post_sync_setup(workspace: Path) -> None:
    """Add site_packages to Python path, clone TRELLIS repo."""
    print()
    print("═" * 60)
    print("  ⚙️  STEP 3 — Post-sync setup")
    print("═" * 60)

    # ── 1. Add site_packages to path ──
    sp_dir = workspace / SITE_PACKAGES_DIR
    if sp_dir.is_dir():
        sp_str = str(sp_dir)
        if sp_str not in sys.path:
            sys.path.insert(0, sp_str)
            print(f"  ✅ Added {SITE_PACKAGES_DIR}/ to sys.path")
        else:
            print(f"  ✅ {SITE_PACKAGES_DIR}/ already in sys.path")
    else:
        print(f"  ⚠️  {SITE_PACKAGES_DIR}/ not found — packages may not be available")

    models_dir = workspace / MODELS_DIR
    if models_dir.is_dir():
        print(f"  ✅ {MODELS_DIR}/ present ({sum(1 for _ in models_dir.rglob('*') if _.is_file())} files)")
    else:
        print(f"  ⚠️  {MODELS_DIR}/ not found — model weights may not be available")

    # ── 2. Clone/update TRELLIS (not a pip package) ──
    trellis_dir = workspace / "TRELLIS"
    if trellis_dir.is_dir():
        print(f"  🔄 Updating TRELLIS repo …")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=trellis_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"  ✅ TRELLIS repo updated")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  TRELLIS git pull failed — will use existing version")
    else:
        print(f"  📥 Cloning TRELLIS repo …")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/microsoft/TRELLIS.git", str(trellis_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"  ✅ TRELLIS repo cloned")
        except subprocess.CalledProcessError as exc:
            print(f"  ❌ TRELLIS clone failed: {exc}")
            print(f"     You may need to clone it manually.")

    # Add TRELLIS to sys.path
    if trellis_dir.is_dir():
        trellis_str = str(trellis_dir)
        if trellis_str not in sys.path:
            sys.path.insert(0, trellis_str)
            print(f"  ✅ Added TRELLIS/ to sys.path")
        else:
            print(f"  ✅ TRELLIS/ already in sys.path")


# ─────────────────────────────────────────────────────────────────────────────
# Final summary & validation
# ─────────────────────────────────────────────────────────────────────────────

def final_summary(workspace: Path) -> None:
    """Print a quick summary of the workspace state."""
    print()
    print("═" * 60)
    print("  📊 WORKSPACE SUMMARY")
    print("═" * 60)

    dirs_to_check = [
        ("Code (from GitHub)", workspace, [".py", ".json", ".md", ".txt", ".yaml"]),
        ("TRELLIS repo",       workspace / "TRELLIS", [".py"]),
        ("Model weights",      workspace / MODELS_DIR, None),
        ("Site packages",      workspace / SITE_PACKAGES_DIR, None),
    ]

    all_ok = True
    for label, path, extensions in dirs_to_check:
        if path.is_dir():
            if extensions:
                count = sum(
                    1 for f in path.rglob("*")
                    if f.is_file() and f.suffix in extensions
                )
            else:
                count = sum(1 for f in path.rglob("*") if f.is_file())
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  ✅ {label}: {count} files, {size / (1024**2):.1f} MiB")
        else:
            print(f"  ❌ {label}: NOT FOUND at {path}")
            all_ok = False

    print()
    if all_ok:
        print("  🎉 Everything is ready! You can now run the pipeline script.")
        print(f"     Open: {workspace / 'videogame_assets_pipeline.py'}")
    else:
        print("  ⚠️  Some components are missing. Re-run this bootstrap to retry.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  🎮 Videogame Assets Generator — Colab Bootstrap           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  GitHub:    {GITHUB_REPO}")
    print(f"  VPS:       {VPS_URL}")
    print(f"  Workspace: {WORKSPACE}")
    print()

    # --- Password prompt (once) ---
    password = getpass.getpass("🔑 Enter VPS password: ")
    if not password:
        print("❌ Password cannot be empty.")
        return

    workspace = Path(WORKSPACE)
    workspace.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # --- Step 1: GitHub sync ---
    try:
        github_sync(GITHUB_REPO, GITHUB_BRANCH, workspace)
    except subprocess.CalledProcessError as exc:
        print(f"  ❌ Git error: {exc}")
        print("  Check GITHUB_REPO URL and that 'git' is available.")
        return
    except Exception as exc:
        print(f"  ❌ GitHub sync failed: {exc}")
        return

    # --- Step 2: VPS sync ---
    try:
        vps_sync(VPS_URL, password, workspace)
    except ConnectionRefusedError as exc:
        print(f"  ❌ {exc}")
        return
    except Exception as exc:
        print(f"  ❌ VPS sync failed: {exc}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 3: Post-sync ---
    post_sync_setup(workspace)

    # --- Summary ---
    final_summary(workspace)

    elapsed = time.time() - t_start
    print(f"\n  ⏱️  Total bootstrap time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
else:
    # When run as a Colab cell (not __main__), execute directly
    main()
