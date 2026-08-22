# Footpath Damage Segmentation
 
A U-Net (EfficientNet-B1 encoder) that segments footpath damage — cracks, potholes, missing pavers — from a photo or video, and scores how severe each detection is.
 
## Demo
 
Run the notebook end-to-end and it will:
1. Load the trained model.
2. Predict on a photo or video you provide.
3. Score severity (low / medium / high) from a mix of damage coverage and crack width.
4. For video, output an annotated clip with a live metadata overlay, plus a structured `issues.json` log of every detection.
## Quickstart
 
1. Clone this repo.
2. Download `best_model.pt` from the [Releases](../../releases) page and place it at `checkpoints/best_model.pt`.
3. Open `footpath_damage_segmentation.ipynb` — in [Colab](https://colab.research.google.com/) or locally with Jupyter — and run all cells.
That's it. No dataset or training setup needed just to get predictions.
 
## Repo contents
 
| File | Purpose |
|---|---|
| `footpath_damage_segmentation.ipynb` | Load the trained model and run it on your own photos/video |
| `checkpoints/best_model.pt` | Trained weights — [download from Releases](../../releases) |
| `train.py` | Train the model from scratch on the labeled dataset |
| `finetune.py` | Fine-tune the checkpoint on a small number of new example images |
 
## Training / fine-tuning
 
You don't need these to run inference with the shipped checkpoint — they're here for anyone who wants to reproduce or extend the model.
 
- **`train.py`** trains from scratch. It expects the dataset laid out as `Train/` and `Val/` folders, each containing `image.jpg` + `image.png` mask pairs.
- **`finetune.py`** takes the existing checkpoint plus a handful of new (image, mask) examples and fine-tunes at a low learning rate, checking after each epoch that accuracy on the original validation set hasn't regressed.
## Model details
 
- Architecture: U-Net, EfficientNet-B1 encoder (ImageNet-pretrained)
- Input size: 512×512
- Trained on the Indian Footpath Damage Segmentation dataset
 
