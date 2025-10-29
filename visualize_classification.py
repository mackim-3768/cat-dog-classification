#!/usr/bin/env python3
"""
Overlay classification results on the input image from ExecuTorch runner output.

Usage examples:
  # 1) Make input_0.bin from an image (224x224, MobileNetV2 norm by default)
  #    python3 make_input_bin.py /path/to/image.jpg --out-dir ./artifacts_from_image
  #    (optionally copy your .pte into the same dir with --copy-pte)
  # 2) Run on device via run_adb_sample.py (will pull outputs under <artifact>/device_out/out)
  #    python3 run_adb_sample.py -a ./artifacts_from_image
  # 3) Visualize Top-K predictions on the original image
  #    python3 visualize_classification.py \
  #        --image /path/to/image.jpg \
  #        --artifact ./artifacts_from_image \
  #        --topk 5 \
  #        --labels /optional/labels.txt

Notes:
- This script assumes the runner outputs FP32 logits or scores in output_0.bin.
- If your model expects a different preprocessing than make_input_bin.py (e.g., different mean/std),
  please generate input_0.bin accordingly so results make sense.
- Labels file is optional; provide one label per line to map class indices.
"""

import argparse
import math
import os
import sys
from array import array
from typing import List

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    print(
        "[ERROR] Pillow (PIL) is required. Install: python -m pip install pillow",
        file=sys.stderr,
    )
    sys.exit(1)


def softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s == 0:
        return [0.0 for _ in xs]
    return [e / s for e in exps]


def load_scores(bin_path: str) -> List[float]:
    nbytes = os.path.getsize(bin_path)
    if nbytes % 4 != 0:
        print(
            f"[WARN] output size {nbytes} is not multiple of 4 bytes; will still try to parse as float32",
            file=sys.stderr,
        )
    buf = array("f")
    with open(bin_path, "rb") as f:
        buf.fromfile(f, nbytes // 4)
    return list(buf)


def load_labels(labels_path: str, n: int) -> List[str]:
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"[WARN] Failed to read labels file: {e}", file=sys.stderr)
        return [f"class_{i}" for i in range(n)]
    if len(lines) < n:
        # Pad if fewer labels than classes
        lines += [f"class_{i}" for i in range(len(lines), n)]
    return lines


def draw_overlay(img: Image.Image, lines: List[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    padding = 6
    line_h = 16

    # Estimate text width
    widths = []
    for t in lines:
        try:
            if hasattr(draw, "textlength"):
                widths.append(draw.textlength(t, font=font))
            else:
                widths.append(len(t) * 7)  # rough fallback
        except Exception:
            widths.append(len(t) * 7)
    box_w = int(max(widths) + 2 * padding) if widths else 200
    box_h = int(line_h * len(lines) + 2 * padding)

    # Draw translucent background
    try:
        draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 180))
    except TypeError:
        # Some PIL versions ignore alpha in RGB mode; draw opaque as fallback
        draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0))

    # Draw text
    y = padding
    for t in lines:
        draw.text((padding, y), t, fill=(255, 255, 255), font=font)
        y += line_h

    return img


def main():
    p = argparse.ArgumentParser(
        description="Overlay classification Top-K results onto the input image."
    )
    p.add_argument(
        "--image",
        required=True,
        help="Original input image used to create input_0.bin",
    )
    p.add_argument(
        "--artifact",
        default=None,
        help="Artifact dir containing device_out/out/output_0.bin",
    )
    p.add_argument(
        "--output-bin",
        default=None,
        help="Override path to output_0.bin if not using --artifact",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="Optional labels file (one label per line)",
    )
    p.add_argument("--topk", type=int, default=5, help="Top-K predictions to show")
    p.add_argument(
        "--out",
        default=None,
        help="Output overlay image path (default: <artifact>/overlay.jpg or ./overlay.jpg)",
    )

    args = p.parse_args()

    if args.output_bin:
        out_bin = args.output_bin
    else:
        if not args.artifact:
            print(
                "Either --output-bin or --artifact must be provided.",
                file=sys.stderr,
            )
            sys.exit(2)
        out_bin = os.path.join(args.artifact, "device_out", "out", "output_0.bin")

    if not os.path.isfile(out_bin):
        print(f"[ERROR] Output bin not found: {out_bin}", file=sys.stderr)
        sys.exit(2)

    scores = load_scores(out_bin)
    if not scores:
        print(f"[ERROR] No scores parsed from {out_bin}", file=sys.stderr)
        sys.exit(2)

    probs = softmax(scores)
    k = max(1, min(args.topk, len(probs)))
    top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]

    if args.labels:
        labels = load_labels(args.labels, len(probs))
    else:
        labels = [f"class_{i}" for i in range(len(probs))]

    lines = [f"{r+1}. {labels[i]}: {probs[i]*100:.2f}%" for r, i in enumerate(top_idx)]

    img = Image.open(args.image).convert("RGB")
    img = draw_overlay(img, lines)

    out_path = (
        args.out
        if args.out
        else (os.path.join(args.artifact, "overlay.jpg") if args.artifact else "overlay.jpg")
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, quality=95)
    print(f"[OK] Wrote overlay: {out_path}")
    print("Top-K:")
    for ln in lines:
        print(" ", ln)


if __name__ == "__main__":
    main()
