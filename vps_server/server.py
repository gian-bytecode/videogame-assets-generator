#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Videogame Assets Generator — VPS File Server                              ║
║  WebSocket Secure (WSS) server with bcrypt authentication                  ║
║                                                                            ║
║  NOT a real SSH server. This is a purpose-built file-distribution daemon    ║
║  that serves/receives files, verifies hashes, and tracks build progress.   ║
║  No shell, no exec, no RCE.                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python server.py                          # uses config.yaml defaults
    python server.py --config /path/to.yaml   # custom config

Protocol (all JSON over WSS, except binary chunks):

  ── Authentication ──
    → {"action":"auth",     "password":"..."}
    ← {"status":"ok",       "session_id":"..."}

  ── Download (server → client) ──
    → {"action":"manifest", "session_id":"..."}
    ← {"status":"ok",       "files":{"rel/path":{"sha256":"...","size":N}, ...}}

    → {"action":"verify",   "session_id":"...", "files":{"rel/path":"sha256",...}}
    ← {"status":"ok",       "to_download":["rel/path1","rel/path2",...]}

    → {"action":"download", "session_id":"...", "path":"rel/path"}
    ← {"status":"ok",       "path":"...", "sha256":"...", "size":N, "total_chunks":N}
    ← <binary chunk 1>  …  <binary chunk N>
    ← {"status":"transfer_complete", "sha256":"..."}

  ── Upload (client → server, zip-based) ──
    → {"action":"upload_verify","session_id":"...", "files":{"rel/path":"sha256",...}}
    ← {"status":"ok",          "to_upload":["rel/path1",...]}

    → {"action":"upload_zip_begin", "session_id":"...", "sha256":"...", "size":N}
    ← {"status":"ok",              "ready":true}
    → <binary chunk 1>  …  <binary chunk N>
    → {"action":"upload_zip_done",  "session_id":"...", "sha256":"..."}
    ← {"status":"ok",              "verified":true, "extracted":N}

  ── Build Progress ──
    → {"action":"get_progress",  "session_id":"..."}
    ← {"status":"ok",           "completed_packages":[...], "total":N}

    → {"action":"mark_package", "session_id":"...", "package_name":"..."}
    ← {"status":"ok"}

    → {"action":"reset_progress","session_id":"..."}
    ← {"status":"ok"}
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import ssl
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import bcrypt
import websockets
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG = logging.getLogger("vps-server")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s"))
LOG.addHandler(_handler)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB per WebSocket frame
SESSION_TTL = 3600            # 1 hour
MAX_SESSIONS = 32             # prevent DoS from filling RAM

# ──────────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────────
SESSIONS: dict[str, float] = {}   # session_id → creation timestamp
CONFIG: dict[str, Any] = {}
PASSWORD_HASH: bytes = b""
FILE_ROOT: Path = Path(".")
PROGRESS_FILE: Path = Path("build_progress.json")

# Per-connection upload state: ws_id → {path, sha256, size, received, hasher, fh}
UPLOAD_STATE: dict[int, dict[str, Any]] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file (streaming, no full-file read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)  # 8 MiB blocks
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_manifest() -> dict[str, dict[str, Any]]:
    """Walk FILE_ROOT and return {relative_path: {sha256, size}} for every file."""
    manifest: dict[str, dict[str, Any]] = {}
    for root, _dirs, files in os.walk(FILE_ROOT):
        _dirs[:] = [d for d in _dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."):
                continue
            full = Path(root) / fname
            rel = full.relative_to(FILE_ROOT).as_posix()
            try:
                stat = full.stat()
                manifest[rel] = {
                    "sha256": sha256_file(full),
                    "size": stat.st_size,
                }
            except OSError as exc:
                LOG.warning("Cannot stat %s: %s", rel, exc)
    return manifest


def prune_sessions() -> None:
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, ts in SESSIONS.items() if now - ts > SESSION_TTL]
    for sid in expired:
        del SESSIONS[sid]


def validate_session(sid: str | None) -> bool:
    """Check if a session_id is valid and alive."""
    if not sid or sid not in SESSIONS:
        return False
    if time.time() - SESSIONS[sid] > SESSION_TTL:
        del SESSIONS[sid]
        return False
    return True


def safe_resolve(requested: str, must_exist: bool = True) -> Path | None:
    """
    Resolve a client-requested path, preventing directory traversal.
    Returns the absolute Path if safe, None otherwise.
    If must_exist=False, the file doesn't need to exist yet (for uploads).
    """
    try:
        # Reject obvious traversal
        if ".." in requested.split("/"):
            return None
        target = (FILE_ROOT / requested).resolve()
        root_resolved = FILE_ROOT.resolve()
        if not str(target).startswith(str(root_resolved) + os.sep) and target != root_resolved:
            return None
        if target == root_resolved:
            return None
        if must_exist and not target.is_file():
            return None
        return target
    except Exception:
        return None


# ── Build Progress Persistence ──

def load_progress() -> list[str]:
    """Load completed package names from disk."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text("utf-8"))
            return data.get("completed_packages", [])
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_progress(completed: list[str]) -> None:
    """Persist completed package names to disk."""
    PROGRESS_FILE.write_text(
        json.dumps({"completed_packages": completed}, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Handlers — Authentication
# ══════════════════════════════════════════════════════════════════════════════

async def handle_auth(ws: Any, data: dict) -> None:
    password = data.get("password", "")
    if not password:
        await ws.send(json.dumps({"status": "error", "message": "Password required"}))
        return

    if bcrypt.checkpw(password.encode("utf-8"), PASSWORD_HASH):
        prune_sessions()
        if len(SESSIONS) >= MAX_SESSIONS:
            await ws.send(json.dumps({
                "status": "error",
                "message": "Too many active sessions, try later"
            }))
            return
        session_id = secrets.token_hex(32)
        SESSIONS[session_id] = time.time()
        LOG.info("✅ Auth OK → session %s…", session_id[:12])
        await ws.send(json.dumps({"status": "ok", "session_id": session_id}))
    else:
        LOG.warning("❌ Auth FAILED from %s", ws.remote_address)
        await asyncio.sleep(1)
        await ws.send(json.dumps({"status": "error", "message": "Invalid password"}))


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Handlers — Download (server → client)
# ══════════════════════════════════════════════════════════════════════════════

async def handle_manifest(ws: Any, data: dict) -> None:
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    LOG.info("📋 Building manifest …")
    manifest = build_manifest()
    LOG.info("📋 Manifest: %d files", len(manifest))
    await ws.send(json.dumps({"status": "ok", "files": manifest}))


async def handle_verify(ws: Any, data: dict) -> None:
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    client_files: dict[str, str] = data.get("files", {})
    server_manifest = build_manifest()

    to_download: list[str] = []
    for rel_path, info in server_manifest.items():
        client_hash = client_files.get(rel_path)
        if client_hash != info["sha256"]:
            to_download.append(rel_path)

    LOG.info("🔍 Verify: %d/%d files need update", len(to_download), len(server_manifest))
    await ws.send(json.dumps({"status": "ok", "to_download": to_download}))


async def handle_download(ws: Any, data: dict) -> None:
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    requested = data.get("path", "")
    target = safe_resolve(requested)
    if target is None:
        await ws.send(json.dumps({
            "status": "error",
            "message": f"File not found or access denied: {requested}"
        }))
        return

    file_size = target.stat().st_size
    file_hash = sha256_file(target)
    total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    if total_chunks == 0:
        total_chunks = 1

    await ws.send(json.dumps({
        "status": "ok",
        "path": requested,
        "sha256": file_hash,
        "size": file_size,
        "total_chunks": total_chunks,
    }))

    with open(target, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            await ws.send(chunk)

    await ws.send(json.dumps({
        "status": "transfer_complete",
        "path": requested,
        "sha256": file_hash,
    }))
    LOG.info("📤 Sent %s (%s bytes, %d chunks)", requested, f"{file_size:,}", total_chunks)


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Handlers — Upload (client → server)
# ══════════════════════════════════════════════════════════════════════════════

async def handle_upload_zip_begin(ws: Any, data: dict) -> None:
    """Prepare to receive a zip archive from the client."""
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    expected_sha = data.get("sha256", "")
    expected_size = data.get("size", 0)

    if not expected_sha:
        await ws.send(json.dumps({"status": "error", "message": "sha256 required"}))
        return

    # Temp file inside FILE_ROOT for the incoming zip
    tmp_path = FILE_ROOT / f".upload_{secrets.token_hex(8)}.zip"
    fh = open(tmp_path, "wb")

    ws_id = id(ws)
    UPLOAD_STATE[ws_id] = {
        "type": "zip",
        "path": "<zip>",
        "tmp_path": tmp_path,
        "expected_sha": expected_sha,
        "expected_size": expected_size,
        "received": 0,
        "hasher": hashlib.sha256(),
        "fh": fh,
    }

    LOG.info("📥 Zip upload begin: %s bytes expected", f"{expected_size:,}")
    await ws.send(json.dumps({"status": "ok", "ready": True}))


async def handle_upload_chunk(ws: Any, chunk: bytes) -> None:
    """Handle a binary chunk during an active upload."""
    ws_id = id(ws)
    state = UPLOAD_STATE.get(ws_id)
    if state is None:
        await ws.send(json.dumps({
            "status": "error",
            "message": "No active upload — send upload_begin first"
        }))
        return

    state["fh"].write(chunk)
    state["hasher"].update(chunk)
    state["received"] += len(chunk)


async def handle_upload_zip_done(ws: Any, data: dict) -> None:
    """Finalize zip upload: verify hash, extract all files into FILE_ROOT."""
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    ws_id = id(ws)
    state = UPLOAD_STATE.pop(ws_id, None)
    if state is None:
        await ws.send(json.dumps({
            "status": "error",
            "message": "No active upload to finalize"
        }))
        return

    state["fh"].close()

    actual_sha = state["hasher"].hexdigest()
    expected_sha = data.get("sha256", state["expected_sha"])
    tmp_path: Path = state["tmp_path"]

    if actual_sha != expected_sha:
        tmp_path.unlink(missing_ok=True)
        LOG.warning("❌ Zip hash mismatch (expected %s, got %s)",
                    expected_sha[:16], actual_sha[:16])
        await ws.send(json.dumps({
            "status": "error",
            "message": "Zip hash mismatch",
            "expected": expected_sha,
            "actual": actual_sha,
        }))
        return

    # Extract zip into FILE_ROOT
    extracted = 0
    root_resolved = FILE_ROOT.resolve()
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Security: reject path-traversal attempts
                member = Path(info.filename)
                if ".." in member.parts:
                    LOG.warning("⚠️  Skipping traversal attempt: %s", info.filename)
                    continue
                target = (FILE_ROOT / info.filename).resolve()
                if not str(target).startswith(str(root_resolved) + os.sep) and target != root_resolved:
                    LOG.warning("⚠️  Skipping escape attempt: %s", info.filename)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
    except zipfile.BadZipFile:
        tmp_path.unlink(missing_ok=True)
        LOG.error("❌ Bad zip file received (%s bytes)", f"{state['received']:,}")
        await ws.send(json.dumps({"status": "error", "message": "Invalid zip file"}))
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    LOG.info("📥 Zip extracted: %d files (%s bytes, sha256:%s…)",
             extracted, f"{state['received']:,}", actual_sha[:16])
    await ws.send(json.dumps({
        "status": "ok",
        "verified": True,
        "extracted": extracted,
    }))


async def handle_upload_verify(ws: Any, data: dict) -> None:
    """Client sends its local hashes; server replies with which files need uploading."""
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    client_files: dict[str, str] = data.get("files", {})
    server_manifest = build_manifest()

    to_upload: list[str] = []
    for rel_path, client_hash in client_files.items():
        server_info = server_manifest.get(rel_path)
        if server_info is None or server_info["sha256"] != client_hash:
            to_upload.append(rel_path)

    # Also flag files that client has but server doesn't
    # (already covered above since server_info would be None)

    LOG.info("🔍 Upload verify: %d/%d files need uploading", len(to_upload), len(client_files))
    await ws.send(json.dumps({"status": "ok", "to_upload": to_upload}))


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Handlers — Build Progress
# ══════════════════════════════════════════════════════════════════════════════

async def handle_get_progress(ws: Any, data: dict) -> None:
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    completed = load_progress()
    LOG.info("📊 Progress: %d packages completed", len(completed))
    await ws.send(json.dumps({
        "status": "ok",
        "completed_packages": completed,
        "total": len(completed),
    }))


async def handle_mark_package(ws: Any, data: dict) -> None:
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    pkg = data.get("package_name", "")
    if not pkg:
        await ws.send(json.dumps({"status": "error", "message": "package_name required"}))
        return

    completed = load_progress()
    if pkg not in completed:
        completed.append(pkg)
        save_progress(completed)
        LOG.info("✅ Package marked complete: %s (total: %d)", pkg, len(completed))
    else:
        LOG.info("ℹ️  Package already marked: %s", pkg)

    await ws.send(json.dumps({"status": "ok", "package_name": pkg}))


async def handle_reset_progress(ws: Any, data: dict) -> None:
    if not validate_session(data.get("session_id")):
        await ws.send(json.dumps({"status": "error", "message": "Invalid session"}))
        return

    save_progress([])
    LOG.info("🔄 Build progress reset")
    await ws.send(json.dumps({"status": "ok"}))


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Handler
# ══════════════════════════════════════════════════════════════════════════════

HANDLERS = {
    "auth": handle_auth,
    # Download
    "manifest": handle_manifest,
    "verify": handle_verify,
    "download": handle_download,
    # Upload
    "upload_zip_begin": handle_upload_zip_begin,
    "upload_zip_done": handle_upload_zip_done,
    "upload_verify": handle_upload_verify,
    # Progress
    "get_progress": handle_get_progress,
    "mark_package": handle_mark_package,
    "reset_progress": handle_reset_progress,
}


async def connection_handler(ws: Any) -> None:
    peer = ws.remote_address
    LOG.info("🔌 Connection from %s", peer)
    ws_id = id(ws)
    try:
        async for message in ws:
            # Binary messages are upload chunks
            if isinstance(message, bytes):
                if ws_id in UPLOAD_STATE:
                    await handle_upload_chunk(ws, message)
                else:
                    await ws.send(json.dumps({
                        "status": "error",
                        "message": "Binary received but no active upload"
                    }))
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await ws.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON"
                }))
                continue

            action = data.get("action")
            handler = HANDLERS.get(action)  # type: ignore[arg-type]
            if handler is None:
                await ws.send(json.dumps({
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }))
                continue

            await handler(ws, data)

    except websockets.exceptions.ConnectionClosed:
        LOG.info("🔌 Connection closed: %s", peer)
    except Exception as exc:
        LOG.exception("💥 Unexpected error from %s: %s", peer, exc)
    finally:
        # Cleanup any abandoned upload
        state = UPLOAD_STATE.pop(ws_id, None)
        if state:
            try:
                state["fh"].close()
            except Exception:
                pass
            tmp_path = state.get("tmp_path")
            if tmp_path and isinstance(tmp_path, Path):
                tmp_path.unlink(missing_ok=True)
            LOG.warning("🗑️  Cleaned up abandoned upload for %s", state.get("path"))


# ══════════════════════════════════════════════════════════════════════════════
# Server Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_password_hash(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read().strip()


async def main() -> None:
    global CONFIG, PASSWORD_HASH, FILE_ROOT, PROGRESS_FILE

    parser = argparse.ArgumentParser(description="VPS File Server (WSS + bcrypt)")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load config
    config_file = Path(args.config)
    if not config_file.exists():
        LOG.error("Config file not found: %s", config_file)
        sys.exit(1)
    CONFIG = load_config(str(config_file))

    # Load password hash
    pw_file = Path(CONFIG.get("password_hash_file", "password.hash"))
    if not pw_file.exists():
        LOG.error("Password hash file not found: %s", pw_file)
        LOG.error("Run:  python setup_password.py")
        sys.exit(1)
    PASSWORD_HASH = load_password_hash(str(pw_file))
    LOG.info("🔑 Password hash loaded from %s", pw_file)

    # File root (directory to serve AND receive uploads into)
    FILE_ROOT = Path(CONFIG.get("file_root", "./serve")).resolve()
    if not FILE_ROOT.is_dir():
        FILE_ROOT.mkdir(parents=True, exist_ok=True)
        LOG.info("📂 Created file root: %s", FILE_ROOT)
    LOG.info("📂 Serving/receiving files: %s", FILE_ROOT)

    # Progress file
    progress_path = CONFIG.get("progress_file", "build_progress.json")
    PROGRESS_FILE = Path(progress_path)
    LOG.info("📊 Progress file: %s", PROGRESS_FILE)

    # SSL context (required for WSS)
    ssl_ctx: ssl.SSLContext | None = None
    ssl_cert = CONFIG.get("ssl_cert")
    ssl_key = CONFIG.get("ssl_key")
    if ssl_cert and ssl_key:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(ssl_cert, ssl_key)
        LOG.info("🔒 TLS enabled (cert: %s)", ssl_cert)
    else:
        LOG.warning("⚠️  No SSL configured — running unencrypted WS (not WSS)")
        LOG.warning("   This is fine for testing, but use TLS in production!")

    host = CONFIG.get("host", "0.0.0.0")
    port = CONFIG.get("port", 8765)

    # Global settings
    global CHUNK_SIZE, SESSION_TTL, MAX_SESSIONS
    CHUNK_SIZE = CONFIG.get("chunk_size", CHUNK_SIZE)
    SESSION_TTL = CONFIG.get("session_ttl", SESSION_TTL)
    MAX_SESSIONS = CONFIG.get("max_sessions", MAX_SESSIONS)

    # Max message size — must accommodate upload chunks
    max_msg_size = CONFIG.get("max_message_size", 10 * 1024 * 1024)  # 10 MiB

    LOG.info("🚀 Starting server on %s:%d …", host, port)

    async with websockets.serve(
        connection_handler,
        host,
        port,
        ssl=ssl_ctx,
        max_size=max_msg_size,
        ping_interval=30,
        ping_timeout=60,
    ):
        LOG.info("✅ Server ready. Ctrl+C to stop.")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("🛑 Shutting down.")
