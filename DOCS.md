# Privacy-Preserving Skin Cancer Classification with FHE

This application classifies dermoscopic skin lesion images into 7 diagnostic categories using a neural network that can run under **Fully Homomorphic Encryption (FHE)** — meaning inference can be performed on encrypted patient images without ever decrypting them.

## How It Works

### Model Architecture (`model.py`)

The core model is **QuantVGG11Patch**, a quantized (5-bit) variant of VGG11 built with [Brevitas](https://github.com/Xilinx/brevitas) that operates on 32×32 image patches. Key adaptations for FHE compatibility:

- All `Conv2d` layers replaced with `QuantConv2d` (5-bit weights)
- All `ReLU` replaced with `QuantReLU` (5-bit activations)
- All `MaxPool2d` replaced with `AvgPool2d` (max operations are expensive in FHE)
- Classifier simplified to a single `QuantLinear(512, 7)` layer
- Forward pass is pure sequential with no control flow

**PatchAggregator** is a plain Python wrapper that:
1. Splits a 224×224 image into a 7×7 grid of 49 patches (32×32 each)
2. Runs each patch through `QuantVGG11Patch`
3. Takes the max logit per class across all patches

The aggregation step runs in plaintext, outside the FHE circuit.

### Training (`train.py`)

Fine-tunes the model on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) skin cancer dataset (10,015 dermoscopic images, 7 classes):

| Class | Description |
|-------|-------------|
| akiec | Actinic keratoses / intraepithelial carcinoma |
| bcc   | Basal cell carcinoma |
| bkl   | Benign keratosis-like lesions |
| df    | Dermatofibroma |
| mel   | Melanoma |
| nv    | Melanocytic nevi |
| vasc  | Vascular lesions |

Features:
- Initializes conv layers from pretrained ImageNet VGG11 weights
- Splits train/val by `lesion_id` to prevent data leakage
- Uses inverse-frequency class weighting to handle imbalance
- Supports Apple Metal (MPS) and CPU

```
python train.py --data-dir ./data --epochs 20 --batch-size 4 --lr 1e-4
```

### FHE Conversion (`fhe_convert.py`)

Compiles the trained patch model into an FHE circuit using [Concrete-ML](https://github.com/zama-ai/concrete-ml):

1. Loads trained weights
2. Builds calibration data (random patches from the validation set)
3. Compiles `QuantVGG11Patch` to an FHE circuit via `compile_brevitas_qat_model`
4. Evaluates accuracy using FHE simulation mode
5. Optionally sweeps `rounding_threshold_bits` (8 → 4) to find speed/accuracy tradeoffs

```
python fhe_convert.py --data-dir ./data --model-path quant_vgg11_patch.pth
python fhe_convert.py --data-dir ./data --model-path quant_vgg11_patch.pth --sweep-rounding
```

## Data Setup

Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it as:

```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

## Dependencies

```
pip install -r requirements.txt
```

Key libraries: `concrete-ml`, `torch`, `torchvision`, `brevitas`, `pandas`, `scikit-learn`, `Pillow`

## Pipeline Summary

```
HAM10000 images
       │
       ▼
  train.py          ──►  quant_vgg11_patch.pth   (trained quantized model)
       │
       ▼
  fhe_convert.py    ──►  FHE circuit             (encrypted inference ready)
       │
       ▼
  Encrypted inference: patient image stays encrypted end-to-end
```
