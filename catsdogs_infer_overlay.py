#!/usr/bin/env python3
"""
End-to-end inference and visualization for Cats vs Dogs model on Samsung E9955.

Steps:
  1) Build input_0.bin from a user image (MobileNetV2 preprocessing)
  2) Push and run on device via adb using enn_executor_runner (through run_adb_sample.py)
  3) Overlay Top-K predictions on the original image

Example:
  python3 catsdogs_infer_overlay.py \
    --image ./cats_and_dogs_filtered/validation/cats/cat.2000.jpg \
    --model ./artifacts_catsdogs/catsdogs_mobilenetv2.pte \
    --labels ./artifacts_catsdogs/labels.txt \
    --artifact ./artifacts_catsdogs_infer \
    --runner ../executorch/build_samsung_android/backends/samsung/enn_executor_runner \
    --serial <device_serial> \
    --topk 2
"""

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional


HERE = os.path.dirname(__file__)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run(cmd: List[str], check: bool = True, capture: bool = False, text: bool = True, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=check, text=text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    return subprocess.run(cmd, check=check, cwd=cwd)


essential_scripts = [
    os.path.join(HERE, "make_input_bin.py"),
    os.path.join(HERE, "run_adb_sample.py"),
    os.path.join(HERE, "visualize_classification.py"),
]


def main():
    p = argparse.ArgumentParser(description="Cats vs Dogs on-device inference with overlay")
    p.add_argument("--image", required=True, type=str, help="Path to a cat/dog image (RGB)")
    p.add_argument("--model", default=os.path.join(HERE, "artifacts_catsdogs", "catsdogs_mobilenetv2.pte"), type=str, help="Path to .pte model")
    p.add_argument("--labels", default=os.path.join(HERE, "artifacts_catsdogs", "labels.txt"), type=str, help="Labels file with two lines: cat, dog")
    p.add_argument("--artifact", default=os.path.join(HERE, "artifacts_catsdogs_infer"), type=str, help="Working dir to store input/output and overlay")
    p.add_argument("--runner", default=os.path.join(HERE, "..", "executorch", "build_samsung_android", "backends", "samsung", "enn_executor_runner"), type=str, help="Path to enn_executor_runner (Android)")
    p.add_argument("--serial", default=None, type=str, help="ADB device serial (optional)")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--topk", type=int, default=2)

    args = p.parse_args()

    # Sanity checks
    for s in essential_scripts:
        if not os.path.isfile(s):
            print(f"[ERROR] Required script not found: {s}", file=sys.stderr)
            sys.exit(2)

    if not which("adb"):
        print("[ERROR] adb not found in PATH", file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model (.pte) not found: {args.model}", file=sys.stderr)
        print("Run training/export first: catsdogs_train_export.py", file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.artifact, exist_ok=True)

    # 1) Make input_0.bin from image (and copy .pte into artifact)
    make_cmd = [
        sys.executable,
        os.path.join(HERE, "make_input_bin.py"),
        args.image,
        "--out-dir", args.artifact,
        "--height", str(args.height),
        "--width", str(args.width),
        "--preset", "mobilenet_v2",
        "--copy-pte", args.model,
    ]
    print("[STEP] Generating input_0.bin from image")
    run(make_cmd)

    # 2) Run on device via adb
    adb_cmd = [
        sys.executable,
        os.path.join(HERE, "run_adb_sample.py"),
        "-a", args.artifact,
        "-r", os.path.abspath(args.runner),
    ]
    if args.serial:
        adb_cmd.extend(["-s", args.serial])
    print("[STEP] Running inference on device via adb")
    run(adb_cmd)

    # 3) Overlay results on the original image
    viz_cmd = [
        sys.executable,
        os.path.join(HERE, "visualize_classification.py"),
        "--image", args.image,
        "--artifact", args.artifact,
        "--labels", args.labels,
        "--topk", str(args.topk),
        "--out", os.path.join(args.artifact, "overlay.jpg"),
    ]
    print("[STEP] Creating overlay image")
    run(viz_cmd)

    print("\n[OK] Completed on-device inference and overlay.")
    print(f"- Artifact: {args.artifact}")
    print(f"- Overlay : {os.path.join(args.artifact, 'overlay.jpg')}")


if __name__ == "__main__":
    main()
