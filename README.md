# Footpath Damage Segmentation

A U-Net (EfficientNet-B1 encoder) that segments footpath damage — cracks, potholes, missing pavers — from a photo or video, scores severity, and optionally geotags each detection.

## Quickstart

1. Clone this repo.
2. Download `best_model.pt` from the [Releases](../../releases) page and put it at `checkpoints/best_model.pt`.
3. Open `footpath_damage_segmentation.ipynb` (locally, or in Colab) and run all cells.

No training or dataset setup needed to just get predictions — the notebook only needs the checkpoint.

## What's in this repo

| File | Purpose |
|---|---|
| `footpath_damage_segmentation.ipynb` | Load the trained model, run it on your own photos/video, optional test-set check |
| `checkpoints/best_model.pt` | Trained weights (download from Releases, see below) |
| `train.py` | Train the model from scratch on the labeled dataset |
| `finetune.py` | Fine-tune an existing checkpoint on a few new hard-example images |

## Why the checkpoint isn't just committed to the repo

Trained `.pt` files are usually tens to hundreds of MB — too big for a normal git commit (GitHub hard-blocks anything over 100MB, and even 25–50MB bloats every future clone since git never forgets a committed binary). Two ways to handle it, pick whichever matches your file's size:

- **Under ~100MB → GitHub Release.** Simplest option, no extra tooling:
  ```
  gh release create v1.0 checkpoints/best_model.pt --title "v1.0 trained weights" --notes "val_dice=0.53"
  ```
  Then link it in the README (already done above) and in the notebook's intro cell.

- **Any size, or if you want it versioned like a normal file → Git LFS.**
  ```
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes checkpoints/best_model.pt
  git commit -m "Add trained checkpoint via LFS"
  git push
  ```
  Anyone who clones the repo gets the checkpoint automatically (if they have `git-lfs` installed) — no separate download step. Note GitHub's free LFS tier caps at 1GB storage / 1GB bandwidth per month, so this can bite you if the repo gets popular.

Check your file size first: `ls -lh checkpoints/best_model.pt`. If you're not sure which to pick, Releases is the safer default — it costs nothing and doesn't require collaborators to install anything extra.

## Training / fine-tuning

`train.py` trains from scratch on the labeled dataset (expects `Train/` and `Val/` folders of `image.jpg` + `image.png` mask pairs). `finetune.py` takes an existing checkpoint plus a handful of new (image, mask) hard examples and does a low-LR fine-tune with a forgetting check against the original val set, saving a new checkpoint per epoch so you can pick the one that improved without regressing.

Neither is needed just to run inference with the shipped checkpoint.
