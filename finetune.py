"""
Fine-tune an existing checkpoint on a small number of new hard-example
images (things the base model gets wrong) without wrecking what it already
learned on the full training set.

Key design choices:
  1. Low LR (1e-5 vs 1e-4 for base training) so a handful of new images make
     small, targeted corrections rather than overwriting everything.
  2. The hard examples are oversampled (repeated `hard_example_repeat` times
     per epoch, each with a fresh random augmentation) so their signal isn't
     drowned out by the full original training set in a single pass.
  3. Validated against the ORIGINAL val set every epoch. If val_dice drops
     more than `forgetting_threshold` from baseline, that's a warning sign
     of catastrophic forgetting -- inspect before trusting a later epoch.
  4. Writes to a separate output directory -- the source checkpoint is
     never overwritten, so you can always fall back to it.

Usage:
    Edit the hard_examples list in FineTuneConfig (or override it when
    calling main()) with your own (image_path, mask_path) pairs, then:

    python finetune.py
"""
import os
import glob
import random
import json
import time
from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp


@dataclass
class FineTuneConfig:
    # Original data (used only for the forgetting check against val)
    original_base_dir: str = "data/Final Indian Footpath Damage Segmentation Dataset1"

    # (image_path, mask_path) pairs for the new examples you want to correct.
    # Masks should be single-channel, same resolution as the image, 0=background/255=damage.
    hard_examples: tuple = ()

    hard_example_repeat: int = 20  # augmented copies per hard example, per epoch

    img_size: int = 512
    batch_size: int = 8
    num_workers: int = 2
    encoder_name: str = "efficientnet-b1"  # must match the source checkpoint

    epochs: int = 10
    lr: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 42

    source_ckpt: str = "checkpoints/best_model.pt"
    output_ckpt_dir: str = "checkpoints_finetuned"

    forgetting_threshold: float = 0.05  # warn if orig val_dice drops more than this
    strict_shape_check: bool = False    # True = hard-fail on image/mask size mismatch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0)),
        A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(p=0.2),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                         hole_width_range=(0.05, 0.15), p=0.3),
        A.GridDistortion(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_tf = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
    return train_tf, val_tf


def load_pairs(split_dir: str) -> list:
    jpgs = sorted(glob.glob(os.path.join(split_dir, "*.jpg")))
    return [(jp, jp[:-4] + ".png") for jp in jpgs if os.path.exists(jp[:-4] + ".png")]


def diagnose_empty_dir(split_dir: str) -> str:
    if not os.path.isdir(split_dir):
        parent = os.path.dirname(split_dir.rstrip("/"))
        parent_listing = os.listdir(parent) if os.path.isdir(parent) else f"(parent '{parent}' also does not exist)"
        return f"Directory does not exist: {split_dir}\nContents of parent directory instead: {parent_listing}"
    all_files = os.listdir(split_dir)
    jpgs = [f for f in all_files if f.lower().endswith(".jpg")]
    pngs = [f for f in all_files if f.lower().endswith(".png")]
    return (f"Directory exists but yielded 0 (image, mask) pairs: {split_dir}\n"
            f"Found {len(all_files)} total files, {len(jpgs)} .jpg, {len(pngs)} .png.\n"
            f"First 10 files: {all_files[:10]}\n"
            f"load_pairs() expects each X.jpg to have a matching X.png in the same folder.")


def validate_and_fix_pairs(pairs: list, name: str, strict: bool) -> list:
    """Check every pair's image/mask dimensions match before training starts.
    Mismatches are auto-resized (mask -> image size, nearest-neighbor) unless strict=True."""
    fixed_pairs, problems = [], []
    for img_path, mask_path in pairs:
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            problems.append(f"  UNREADABLE IMAGE: {img_path}")
            continue
        if mask is None:
            problems.append(f"  UNREADABLE MASK: {mask_path}")
            continue
        ih, iw = image.shape[:2]
        mh, mw = mask.shape[:2]
        if (ih, iw) != (mh, mw):
            problems.append(f"  SHAPE MISMATCH: {os.path.basename(img_path)} is {iw}x{ih} "
                             f"but {os.path.basename(mask_path)} is {mw}x{mh}")
            if strict:
                continue
            mask_resized = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
            fixed_mask_path = mask_path.rsplit(".", 1)[0] + "_resized.png"
            cv2.imwrite(fixed_mask_path, mask_resized)
            fixed_pairs.append((img_path, fixed_mask_path))
            print(f"  -> auto-resized mask to {iw}x{ih}, saved to {fixed_mask_path}")
        else:
            fixed_pairs.append((img_path, mask_path))

    if problems:
        print(f"\n[{name}] {len(problems)} issue(s) found among {len(pairs)} pairs:")
        for p in problems:
            print(p)
        if strict and any("SHAPE MISMATCH" in p for p in problems):
            raise ValueError(f"[{name}] strict_shape_check=True and shape mismatches were found.")

    if len(fixed_pairs) == 0:
        raise ValueError(f"[{name}] 0 usable pairs after validation (started with {len(pairs)}).")

    print(f"[{name}] {len(fixed_pairs)}/{len(pairs)} pairs OK for training.")
    return fixed_pairs


class MixedDataset(Dataset):
    """Original training pairs + oversampled hard examples, same augmentation pipeline."""

    def __init__(self, original_pairs: list, hard_examples: tuple, hard_repeat: int, transform):
        self.transform = transform
        self.pairs = list(original_pairs)
        for pair in hard_examples:
            self.pairs.extend([pair] * hard_repeat)
        random.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype("float32")
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0)


class EvalDataset(Dataset):
    def __init__(self, pairs: list, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype("float32")
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


@torch.no_grad()
def compute_dice(pred_logits, target, thresh=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice.mean().item()


def run_epoch(model, loader, criterion, optimizer, scaler, device, train_mode, use_amp):
    model.train() if train_mode else model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc="train" if train_mode else "val", leave=False):
            images, masks = images.to(device), masks.to(device)
            if train_mode:
                optimizer.zero_grad()
                if use_amp:
                    with torch.autocast(device_type="cuda"):
                        preds = model(images)
                        loss = criterion(preds, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(images)
                    loss = criterion(preds, masks)
                    loss.backward()
                    optimizer.step()
            else:
                preds = model(images)
                loss = criterion(preds, masks)
            total_loss += loss.item()
            total_dice += compute_dice(preds.detach(), masks)
            n += 1
    return total_loss / n, total_dice / n


def fine_tune(cfg: FineTuneConfig):
    assert len(cfg.hard_examples) > 0, (
        "cfg.hard_examples is empty. Add (image_path, mask_path) tuples for the "
        "new examples you want the model to learn from."
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print("Device:", device)
    if device == "cpu":
        print("WARNING: running on CPU -- this will be slow for several epochs over the full training set.")

    os.makedirs(cfg.output_ckpt_dir, exist_ok=True)

    model = smp.Unet(encoder_name=cfg.encoder_name, encoder_weights=None,
                      in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(cfg.source_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    baseline_dice_display = ckpt.get("val_dice", ckpt.get("orig_val_dice"))
    print(f"Loaded source checkpoint (val_dice={baseline_dice_display:.4f}) -- fine-tuning from here")

    train_tf, val_tf = build_transforms(cfg.img_size)

    original_train_dir = os.path.join(cfg.original_base_dir, "Train")
    original_val_dir = os.path.join(cfg.original_base_dir, "Val")
    original_train_pairs = load_pairs(original_train_dir)
    original_val_pairs = load_pairs(original_val_dir)

    if len(original_train_pairs) == 0:
        raise FileNotFoundError(
            "original train pairs = 0 -- refusing to fine-tune with no forgetting protection.\n"
            + diagnose_empty_dir(original_train_dir)
        )
    if len(original_val_pairs) == 0:
        raise FileNotFoundError(
            "original val pairs = 0 -- refusing to fine-tune with no forgetting check.\n"
            + diagnose_empty_dir(original_val_dir)
        )

    print(f"Original train: {len(original_train_pairs)} | Original val (forgetting check): {len(original_val_pairs)}")
    print(f"Hard examples: {len(cfg.hard_examples)}, oversampled {cfg.hard_example_repeat}x each per epoch")

    original_train_pairs = validate_and_fix_pairs(original_train_pairs, "original_train", cfg.strict_shape_check)
    original_val_pairs = validate_and_fix_pairs(original_val_pairs, "original_val", cfg.strict_shape_check)
    hard_examples = tuple(validate_and_fix_pairs(list(cfg.hard_examples), "hard_examples", cfg.strict_shape_check))

    train_ds = MixedDataset(original_train_pairs, hard_examples, cfg.hard_example_repeat, train_tf)
    val_ds = EvalDataset(original_val_pairs, val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=(device == "cuda"))

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    baseline_val_dice = ckpt.get("val_dice", ckpt.get("orig_val_dice"))
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, use_amp)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, use_amp)
        elapsed = time.time() - t0

        drop = baseline_val_dice - val_dice
        warning = ""
        if drop > cfg.forgetting_threshold:
            warning = f"  <-- WARNING: original val_dice dropped {drop:.4f} from baseline {baseline_val_dice:.4f}, possible forgetting"

        print(f"Epoch {epoch}/{cfg.epochs} ({elapsed:.0f}s) | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"orig_val_loss={val_loss:.4f} orig_val_dice={val_dice:.4f}{warning}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_dice": train_dice,
                         "orig_val_loss": val_loss, "orig_val_dice": val_dice})

        ckpt_path = os.path.join(cfg.output_ckpt_dir, f"finetuned_epoch{epoch}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "orig_val_dice": val_dice, "baseline_val_dice": baseline_val_dice}, ckpt_path)

    with open(os.path.join(cfg.output_ckpt_dir, "finetune_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Checkpoints saved per-epoch to {cfg.output_ckpt_dir}")
    print(f"Source checkpoint {cfg.source_ckpt} was NOT modified.")
    print("Next: check each finetuned_epochN.pt against your hard examples AND the "
          "original val set, and pick the epoch that fixes the misses without tanking orig_val_dice.")


if __name__ == "__main__":
    cfg = FineTuneConfig(
        hard_examples=(
            # ("path/to/image.jpg", "path/to/mask.png"),
        )
    )
    fine_tune(cfg)
