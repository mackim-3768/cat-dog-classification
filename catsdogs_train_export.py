#!/usr/bin/env python3
"""
Transfer learning on Cats vs Dogs (cats_and_dogs_filtered) with MobileNetV2,
then export to ExecuTorch .pte for Samsung ENN backend.

Dataset layout expected:
  <root>/train/{cats,dogs}
  <root>/validation/{cats,dogs}  # optional; if missing/empty, we split from train

Outputs (under --artifact):
  - catsdogs_mobilenetv2.pte
  - best_ckpt.pth
  - labels.txt  (two lines: cat, dog)

Example:
  python3 catsdogs_train_export.py \
    --data-root ./cats_and_dogs_filtered \
    --artifact ./artifacts_catsdogs \
    --epochs 3 --batch-size 32 --chipset E9955
"""

import argparse
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ExecuTorch Samsung backend imports
from executorch.backends.samsung.partition.enn_partitioner import (
    EnnPartitioner,
)
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    to_edge_transform_and_lower_to_enn,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program


def has_valid_split(root: str) -> bool:
    vdir = os.path.join(root, "validation")
    if not os.path.isdir(vdir):
        return False
    cats = os.path.join(vdir, "cats")
    dogs = os.path.join(vdir, "dogs")
    if not (os.path.isdir(cats) and os.path.isdir(dogs)):
        return False
    # Must not be empty
    def nonempty(d):
        try:
            return any(True for _ in os.scandir(d))
        except FileNotFoundError:
            return False
    return nonempty(cats) and nonempty(dogs)


def build_dataloaders(data_root: str, img_size: int, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    # Use MobileNetV2 default normalization constants explicitly to be robust
    # across torchvision versions.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if has_valid_split(data_root):
        train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
        val_ds = datasets.ImageFolder(os.path.join(data_root, "validation"), transform=val_tf)
    else:
        # Build from train only and split into two independent dataset objects
        # so train/val transforms don't interfere with each other.
        base_dir = os.path.join(data_root, "train")
        ds_train_base = datasets.ImageFolder(base_dir, transform=train_tf)
        ds_val_base = datasets.ImageFolder(base_dir, transform=val_tf)
        n_val = max(1, int(0.2 * len(ds_train_base)))
        indices = torch.randperm(len(ds_train_base)).tolist()
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        train_ds = Subset(ds_train_base, train_indices)
        val_ds = Subset(ds_val_base, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_model(num_classes: int = 2) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def evaluate(model, loader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def export_to_pte(model: nn.Module, artifact: str, chipset: str, h: int, w: int) -> str:
    os.makedirs(artifact, exist_ok=True)
    model_eval = model.eval().cpu()
    example_input = torch.randn(1, 3, h, w, dtype=torch.float32)
    compile_specs = [gen_samsung_backend_compile_spec(chipset)]
    edge_prog = to_edge_transform_and_lower_to_enn(model_eval, (example_input,), compile_specs=compile_specs)
    edge = edge_prog.to_backend(EnnPartitioner(compile_specs))
    exec_prog = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True))
    pte_path = save_pte_program(exec_prog, "catsdogs_mobilenetv2", artifact)
    return pte_path


def main():
    p = argparse.ArgumentParser(description="Cats vs Dogs transfer learning and ExecuTorch export")
    default_root = os.path.join(os.path.dirname(__file__), "cats_and_dogs_filtered")
    p.add_argument("--data-root", default=default_root, type=str, help="Dataset root containing train/ and validation/")
    p.add_argument("--artifact", default=os.path.join(os.path.dirname(__file__), "artifacts_catsdogs"), type=str, help="Output directory for .pte and checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chipset", type=str, default="E9955")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)

    args = p.parse_args()

    for d in [os.path.join(args.data_root, "train")]:
        if not os.path.isdir(d):
            print(f"[ERROR] Dataset not found: {args.data_root}", file=sys.stderr)
            sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader = build_dataloaders(args.data_root, args.height, args.batch_size, args.num_workers)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(args.artifact, exist_ok=True)
    ckpt_path = os.path.join(args.artifact, "best_ckpt.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[E{epoch:02d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "acc": best_acc,
            }, ckpt_path)
            print(f"[INFO] Saved best checkpoint: {ckpt_path} (acc={best_acc:.4f})")

    # Load best and export
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])  # type: ignore[index]
        print(f"[INFO] Loaded best checkpoint (acc={state.get('acc', 0.0):.4f})")

    # Write labels
    labels_txt = os.path.join(args.artifact, "labels.txt")
    with open(labels_txt, "w", encoding="utf-8") as f:
        f.write("cat\n")
        f.write("dog\n")
    print(f"[INFO] Wrote labels: {labels_txt}")

    pte_path = export_to_pte(model, args.artifact, args.chipset, args.height, args.width)
    print("\n[OK] Export complete:")
    print(f"- PTE:   {pte_path}")
    print(f"- CKPT:  {ckpt_path}")
    print(f"- Labels:{labels_txt}")


if __name__ == "__main__":
    main()
