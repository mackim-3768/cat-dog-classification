#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run(cmd: List[str], check: bool = True, capture: bool = False, text: bool = True, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=check, text=text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    return subprocess.run(cmd, check=check, cwd=cwd)


def list_adb_devices() -> List[Tuple[str, str]]:
    cp = run(["adb", "devices"], capture=True)
    lines = [ln.strip() for ln in cp.stdout.strip().splitlines()]
    out: List[Tuple[str, str]] = []
    for ln in lines[1:]:  # skip header
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def ensure_remote_dir(serial: str, remote: str) -> None:
    run(["adb", "-s", serial, "shell", "mkdir", "-p", remote])


def adb_push(serial: str, local: str, remote: str) -> None:
    run(["adb", "-s", serial, "push", local, remote])


def adb_shell(serial: str, cmd: str) -> subprocess.CompletedProcess:
    # Run the command via sh -c for quoting
    return run(["adb", "-s", serial, "shell",  cmd], capture=True)


def pick_device(preferred_serial: Optional[str]) -> str:
    devices = list_adb_devices()
    devices = [d for d in devices if d[1] == "device"]
    if preferred_serial:
        for (serial, state) in devices:
            if serial == preferred_serial:
                return serial
        raise SystemExit(f"No such device: {preferred_serial}")
    if not devices:
        raise SystemExit("No adb devices in 'device' state. Run 'adb devices'.")
    return devices[0][0]


def find_artifacts(artifact_dir: str) -> Tuple[str, str]:
    # Prefer sample_model.pte and input_0.bin
    pte = os.path.join(artifact_dir, "sample_model.pte")
    if not os.path.isfile(pte):
        # fallback to any .pte
        cands = list(Path(artifact_dir).glob("*.pte"))
        if not cands:
            raise SystemExit(f"No .pte under {artifact_dir}")
        pte = str(cands[0])
    inp = os.path.join(artifact_dir, "input_0.bin")
    if not os.path.isfile(inp):
        # fallback to first input_*.bin
        cands = sorted(Path(artifact_dir).glob("input_*.bin"))
        if not cands:
            raise SystemExit(f"No input_*.bin under {artifact_dir}")
        inp = str(cands[0])
    return pte, inp


def main():
    parser = argparse.ArgumentParser(description="Push and run ExecuTorch Samsung sample via adb")
    parser.add_argument(
        "-a",
        "--artifact",
        default=os.path.join(os.path.dirname(__file__), "artifacts_e9955"),
        help="Directory containing exported .pte and inputs",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--runner",
        default=os.path.join(os.path.dirname(__file__), "..", "executorch", "build_samsung_android", "backends", "samsung", "enn_executor_runner"),
        help="Path to enn_executor_runner binary built for Android",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--serial",
        default=None,
        help="ADB device serial (optional). If not set, picks the first 'device'",
        type=str,
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/local/tmp/executorch_sample",
        help="Remote working directory on device",
        type=str,
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Do not pull outputs back from device",
    )

    args = parser.parse_args()

    if not which("adb"):
        print("[ERROR] adb not found in PATH")
        sys.exit(1)

    runner = os.path.abspath(args.runner)
    if not os.path.isfile(runner):
        print(f"[ERROR] Runner not found: {runner}")
        print("Build for Android first, e.g., ensure \"build_samsung_android/backends/samsung/enn_executor_runner\" exists.")
        sys.exit(1)

    artifact_dir = os.path.abspath(args.artifact)
    pte_path, input_bin = find_artifacts(artifact_dir)

    serial = pick_device(args.serial)
    print(f"[INFO] Using device: {serial}")

    remote = args.remote_dir.rstrip("/")
    ensure_remote_dir(serial, remote)
    ensure_remote_dir(serial, f"{remote}/out")

    # Push files
    adb_push(serial, runner, f"{remote}/enn_executor_runner")
    adb_shell(serial, f"chmod 755 {remote}/enn_executor_runner")

    remote_pte = f"{remote}/{os.path.basename(pte_path)}"
    remote_inp = f"{remote}/{os.path.basename(input_bin)}"
    adb_push(serial, pte_path, remote_pte)
    adb_push(serial, input_bin, remote_inp)

    # Run
    cmd = (
        f"{remote}/enn_executor_runner "
        f"--model {remote_pte} "
        f"--input {remote_inp} "
        f"--output_path {remote}/out"
    )

    print("[INFO] Running on device:")
    print("       ", cmd)
    cp = adb_shell(serial, cmd)
    print(cp.stdout)

    if not args.no_pull:
        host_out = os.path.join(artifact_dir, "device_out")
        os.makedirs(host_out, exist_ok=True)
        run(["adb", "-s", serial, "pull", f"{remote}/out", host_out])
        print(f"[OK] Pulled outputs to {host_out}")

    print("[DONE]")


if __name__ == "__main__":
    main()
