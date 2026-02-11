#!/usr/bin/env python3
"""
Utility to create/update the bcrypt password hash for the VPS File Server.

Usage:
    python setup_password.py                     # interactive prompt
    python setup_password.py --output pw.hash    # custom output file
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import bcrypt


def main() -> None:
    parser = argparse.ArgumentParser(description="Set server password (bcrypt hash)")
    parser.add_argument(
        "--output", default="password.hash",
        help="Output file for the bcrypt hash (default: password.hash)"
    )
    parser.add_argument(
        "--rounds", type=int, default=12,
        help="bcrypt cost factor / rounds (default: 12)"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  VPS File Server — Password Setup                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    pw1 = getpass.getpass("Enter new password: ")
    if not pw1:
        print("❌ Password cannot be empty.")
        sys.exit(1)

    pw2 = getpass.getpass("Confirm password:   ")
    if pw1 != pw2:
        print("❌ Passwords do not match.")
        sys.exit(1)

    if len(pw1) < 8:
        print("⚠️  Warning: password is shorter than 8 characters.")
        confirm = input("Continue anyway? [y/N] ").strip().lower()
        if confirm != "y":
            sys.exit(1)

    salt = bcrypt.gensalt(rounds=args.rounds)
    hashed = bcrypt.hashpw(pw1.encode("utf-8"), salt)

    out_path = Path(args.output)
    out_path.write_bytes(hashed + b"\n")

    print()
    print(f"✅ Password hash saved to: {out_path.resolve()}")
    print(f"   bcrypt rounds: {args.rounds}")
    print(f"   Hash: {hashed.decode('utf-8')}")
    print()
    print("⚠️  Do NOT commit this file to git.")
    print("   It should already be in .gitignore.")


if __name__ == "__main__":
    main()
