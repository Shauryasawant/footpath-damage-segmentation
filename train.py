"""
Train the footpath damage segmentation model from scratch (or resume).

Expects the dataset laid out as:
  <base_dir>/Train/*.jpg + *.png   (image, mask pairs, same basename)
  <base_dir>/Val/*.jpg   + *.png

Usage:
    python train.py
Edit the Config dataclass below to point at your dataset / change hyperparameters.
Resumes automatically from ckpt_dir/best_model.pt if it already exists.
"""
import os
import json
import time
import random
import glob
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


@dataclass
class Config:
    base_dir: str = "data/Final Indian Footpath Damage Segmentation Dataset1"
    img_size: int = 512
    batch_size: int = 8
    num_workers: int = 2
    encoder_name: str = "efficientnet-b1"
    encoder_weights: str = "imagenet"
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    seed: int = 42
    ckpt_dir: str = "checkpoints"  # resumes from here if best_model.pt already exists


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(cfg):
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(cfg.img_size, cfg.img_size), scale=(0.6, 1.0)),
        A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(p=0.2),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                         hole_width_range=(0.05, 0.15), p=0.3),
        A.GridDistortion(p=0.2),
        A.Normalize(), ToTensorV2(),
    ])
    val_tf = A.Compose([A.Resize(cfg.img_size, cfg.img_size), A.Normalize(), ToTensorV2()])
    return train_tf, val_tf


def load_pairs(split_dir):
    jpgs = sorted(glob.glob(os.path.join(split_dir, "*.jpg")))
    return [(jp, jp[:-4] + ".png") for jp in jpgs if os.path.exists(jp[:-4] + ".png")]


class FootpathDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype("float32")
        aug = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


def compute_metrics(pred_logits, target, thresh=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return {"dice": dice.mean().item()}


def run_epoch(model, loader, criterion, optimizer, scaler, device, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc="train" if train_mode else "val", leave=False):
            images, masks = images.to(device), masks.to(device)
            if train_mode:
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    preds = model(images)
                    loss = criterion(preds, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss = criterion(preds, masks)
            m = compute_metrics(preds.detach(), masks)
            total_loss += loss.item()
            total_dice += m["dice"]
            n += 1
    return total_loss / n, total_dice / n


def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_path = os.path.join(cfg.ckpt_dir, "best_model.pt")
    history_path = os.path.join(cfg.ckpt_dir, "history.json")

    model = smp.Unet(encoder_name=cfg.encoder_name, encoder_weights=cfg.encoder_weights,
                      in_channels=3, classes=1, activation=None).to(device)

    train_tf, val_tf = build_transforms(cfg)
    train_pairs = load_pairs(os.path.join(cfg.base_dir, "Train"))
    val_pairs = load_pairs(os.path.join(cfg.base_dir, "Val"))
    assert train_pairs, f"No (image, mask) pairs found in {os.path.join(cfg.base_dir, 'Train')}"
    assert val_pairs, f"No (image, mask) pairs found in {os.path.join(cfg.base_dir, 'Val')}"

    train_loader = DataLoader(FootpathDataset(train_pairs, train_tf), batch_size=cfg.batch_size,
                               shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(FootpathDataset(val_pairs, val_tf), batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch, best_val_dice, epochs_no_improve, history = 1, 0.0, 0, []
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_dice = ckpt["val_dice"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {ckpt['epoch']} (val_dice={best_val_dice:.4f})")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
        scheduler.step()
        print(f"Epoch {epoch}/{cfg.epochs} ({time.time()-t0:.0f}s) | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if val_dice > best_val_dice:
            best_val_dice, epochs_no_improve = val_dice, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "val_dice": best_val_dice}, best_path)
            print(f"  -> new best (val_dice={best_val_dice:.4f}), saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print("Early stopping.")
                break

    print(f"Done. Best val_dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main(Config())
