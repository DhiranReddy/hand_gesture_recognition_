#!/usr/bin/env python3
"""
Vendor the MediaPipe runtime so the app does not depend on a CDN at load time.

Without this the app imports ~19 MB of WASM plus a 7.4 MB model from
cdn.jsdelivr.net and storage.googleapis.com on every cold load. That is a
third-party availability dependency and a supply-chain surface on a page that
holds camera access.

The files are too large to commit, so they are fetched on demand and their
SHA-256 digests are recorded in web/vendor/manifest.json, which IS committed.
Re-running verifies against that manifest and fails on a mismatch, so a
tampered or silently updated file is caught.

    python tools/vendor_runtime.py           # fetch and verify
    python tools/vendor_runtime.py --check   # verify what is already there
    python tools/vendor_runtime.py --update  # accept new digests (review them)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "web" / "vendor"
MANIFEST = VENDOR / "manifest.json"

TASKS_VERSION = "0.10.18"
CDN = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}"
MODELS = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1"

FILES = {
    "tasks-vision/vision_bundle.mjs": f"{CDN}/vision_bundle.mjs",
    "tasks-vision/wasm/vision_wasm_internal.js": f"{CDN}/wasm/vision_wasm_internal.js",
    "tasks-vision/wasm/vision_wasm_internal.wasm": f"{CDN}/wasm/vision_wasm_internal.wasm",
    "tasks-vision/wasm/vision_wasm_nosimd_internal.js": f"{CDN}/wasm/vision_wasm_nosimd_internal.js",
    "tasks-vision/wasm/vision_wasm_nosimd_internal.wasm": f"{CDN}/wasm/vision_wasm_nosimd_internal.wasm",
    "hand_landmarker.task": f"{MODELS}/hand_landmarker.task",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        total = int(response.headers.get("content-length", 0))
        done = 0
        with target.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done/1e6:5.1f}/{total/1e6:.1f} MB", end="", flush=True)
    if total:
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor the MediaPipe runtime")
    parser.add_argument("--check", action="store_true", help="Verify without downloading")
    parser.add_argument("--update", action="store_true", help="Record new digests")
    args = parser.parse_args()

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8")).get("files", {})

    failures = []
    recorded = {}

    for name, url in FILES.items():
        target = VENDOR / name
        if not target.exists():
            if args.check:
                print(f"  MISSING  {name}")
                failures.append(name)
                continue
            print(f"  fetching {name}")
            fetch(url, target)

        actual = digest(target)
        expected = manifest.get(name)
        recorded[name] = actual

        if expected is None or args.update:
            print(f"  recorded {name}  {actual[:19]}…")
        elif actual != expected:
            print(f"  MISMATCH {name}\n    expected {expected}\n    actual   {actual}")
            failures.append(name)
        else:
            print(f"  verified {name}")

    if failures:
        print(f"\n{len(failures)} file(s) failed verification.")
        print("If an upstream release is genuinely expected, review the diff and")
        print("re-run with --update to accept the new digests.")
        return 1

    if not args.check:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(
            json.dumps(
                {"tasks_vision_version": TASKS_VERSION, "files": recorded},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        size = sum((VENDOR / n).stat().st_size for n in FILES) / 1e6
        print(f"\nVendored {len(FILES)} files ({size:.0f} MB) into {VENDOR}")
        print("The app prefers these over the CDN automatically.")
    else:
        print("\nAll files verified against the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
