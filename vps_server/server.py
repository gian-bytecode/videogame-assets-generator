#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Videogame Assets Generator — VPS File Server                              ║
║  WebSocket Secure (WSS) server with bcrypt authentication                  ║
║                                                                            ║
║  NOT a real SSH server. This is a purpose-built file-distribution daemon    ║
║  that only serves files and verifies hashes. No shell, no exec, no RCE.    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python server.py                          # uses config.yaml defaults
    python server.py --config /path/to.yaml   # custom config

Protocol (all JSON over WSS, except binary chunks):
    → {"action":"auth",     "password":"..."}
    ← {"status":"ok",       "session_id":"..."}

    → {"action":"manifest", "session_id":"..."}
    ← {"status":"ok",       "files":{"rel/path":{"sha256":"...","size":N}, ...}}

    → {"action":"verify",   "session_id":"...", "files":{"rel/path":"sha256",...}}
    ← {"status":"ok",       "to_download":["rel/path1","rel/path2",...]}

    → {"action":"download", "session_id":"...", "path":"rel/path"}
    ← {"status":"ok",       "path":"...", "sha256":"...", "size":N, "total_chunks":N}
    ← <binary chunk 1>  …  <binary chunk N>
    ← {"status":"transfer_complete", "sha256":"..."}
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import ssl
import sys
import time
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
        # Skip hidden directories (.git, etc.)
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


def safe_resolve(requested: str) -> Path | None:
    """
    Resolve a client-requested path, preventing directory traversal.
    Returns the absolute Path if safe, None otherwise.
    """
    try:
        target = (FILE_ROOT / requested).resolve()
        # Must be under FILE_ROOT
        if FILE_ROOT.resolve() in target.parents or target == FILE_ROOT.resolve():
            return None  # it IS the root or something weird
        if not str(target).startswith(str(FILE_ROOT.resolve())):
            return None  # traversal attempt
        if not target.is_file():
            return None
        return target
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Handlers
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
        # Brief delay to slow brute-force
        await asyncio.sleep(1)
        await ws.send(json.dumps({"status": "error", "message": "Invalid password"}))


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
        total_chunks = 1  # empty file still gets 1 "chunk"

    # Send header
    await ws.send(json.dumps({
        "status": "ok",
        "path": requested,
        "sha256": file_hash,
        "size": file_size,
        "total_chunks": total_chunks,
    }))

    # Stream binary chunks
    sent = 0
    with open(target, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            await ws.send(chunk)
            sent += len(chunk)

    # Send completion with hash for verification
    await ws.send(json.dumps({
        "status": "transfer_complete",
        "path": requested,
        "sha256": file_hash,
    }))
    LOG.info("📤 Sent %s (%s bytes, %d chunks)", requested, f"{file_size:,}", total_chunks)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Handler
# ══════════════════════════════════════════════════════════════════════════════

HANDLERS = {
    "auth": handle_auth,
    "manifest": handle_manifest,
    "verify": handle_verify,
    "download": handle_download,
}


async def connection_handler(ws: Any) -> None:
    peer = ws.remote_address
    LOG.info("🔌 Connection from %s", peer)
    try:
        async for message in ws:
            if isinstance(message, bytes):
                # We don't accept binary from clients
                await ws.send(json.dumps({
                    "status": "error",
                    "message": "Binary messages not accepted"
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
    global CONFIG, PASSWORD_HASH, FILE_ROOT

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

    # File root (directory to serve)
    FILE_ROOT = Path(CONFIG.get("file_root", "./serve")).resolve()
    if not FILE_ROOT.is_dir():
        LOG.error("File root directory not found: %s", FILE_ROOT)
        LOG.error("Create it and place model weights + site_packages inside.")
        sys.exit(1)
    LOG.info("📂 Serving files from: %s", FILE_ROOT)

    # SSL context
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

    # Max message size — client messages are small JSON, but set a sane limit
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
