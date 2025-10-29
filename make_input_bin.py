#!/usr/bin/env python3
"""
Convert an image to ExecuTorch input_0.bin (FP32, shape [1,3,H,W], NCHW, little-endian).
Defaults to MobileNetV2 normalization.

Examples:
  python3 make_input_bin.py /path/to/image.jpg \
      --out-dir ./artifacts_from_image \
      --height 224 --width 224 \
      --preset mobilenet_v2

  # No normalization (just scale to [0,1])
  python3 make_input_bin.py /path/to/image.jpg --preset none

  # Custom mean/std
  python3 make_input_bin.py /path/to/image.jpg \
      --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5

  # Also copy a .pte into the output directory
  python3 make_input_bin.py /path/to/image.jpg --out-dir ./artifacts --copy-pte ./mv2_xnnpack.pte
"""

import argparse
import os
import sys
from array import array
from typing import List

try:
    from PIL import Image
except Exception as e:
    print("[ERROR] Pillow (PIL) is required. Install: python -m pip install pillow", file=sys.stderr)
    sys.exit(1)


def parse_floats_csv(s: str, expected: int) -> List[float]:
    vals = [v for v in s.replace(" ", "").split(",") if v]
    if len(vals) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {len(vals)}: '{s}'")
    return [float(v) for v in vals]


def main():
    parser = argparse.ArgumentParser(description="Make ExecuTorch input_0.bin from an image")
    parser.add_argument("image", type=str, help="Path to input image (RGB)")
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "artifacts_from_image"), help="Output directory; input_0.bin will be written here")
    parser.add_argument("--height", type=int, default=224, help="Resize height")
    parser.add_argument("--width", type=int, default=224, help="Resize width")
    parser.add_argument("--preset", type=str, choices=["mobilenet_v2", "none"], default="mobilenet_v2", help="Normalization preset")
    parser.add_argument("--mean", type=str, default=None, help="Override mean as 'r,g,b' (floats)")
    parser.add_argument("--std", type=str, default=None, help="Override std as 'r,g,b' (floats)")
    parser.add_argument("--copy-pte", type=str, default=None, help="Optional: copy a .pte into out-dir (for convenience)")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load image and resize
    img = Image.open(args.image).convert("RGB").resize((args.width, args.height), Image.BILINEAR)

    # Get channels as flat lists (row-major H*W)
    r, g, b = img.split()
    r_lst = list(r.getdata())
    g_lst = list(g.getdata())
    b_lst = list(b.getdata())

    # Determine normalization
    if args.mean is not None and args.std is not None:
        mean = parse_floats_csv(args.mean, 3)
        std = parse_floats_csv(args.std, 3)
    elif args.preset == "mobilenet_v2":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = None
        std = None

    def to_norm(chan, m=None, s=None):
        if m is None or s is None:  # no normalization; scale to [0,1]
            return [(v / 255.0) for v in chan]
        return [((v / 255.0) - m) / s for v in chan]

    rn = to_norm(r_lst, mean[0], std[0]) if mean is not None else to_norm(r_lst)
    gn = to_norm(g_lst, mean[1], std[1]) if mean is not None else to_norm(g_lst)
    bn = to_norm(b_lst, mean[2], std[2]) if mean is not None else to_norm(b_lst)

    # Pack as FP32 NCHW [1,3,H,W] => C-major (R plane, then G, then B)
    buf = array('f')
    buf.extend(rn)
    buf.extend(gn)
    buf.extend(bn)

    out_path = os.path.join(args.out_dir, "input_0.bin")
    with open(out_path, "wb") as f:
        buf.tofile(f)

    # Optional: copy .pte
    if args.copy_pte:
        import shutil
        shutil.copy2(args.copy_pte, os.path.join(args.out_dir, os.path.basename(args.copy_pte)))

    nbytes = os.path.getsize(out_path)
    nh, nw = args.height, args.width
    print("[OK] Wrote:", out_path)
    print("     shape=[1,3,%d,%d] bytes=%d" % (nh, nw, nbytes))


if __name__ == "__main__":
    main()
