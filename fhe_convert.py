"""
Convert the trained QuantVGG11Patch to an FHE circuit via Concrete-ML.

Steps:
  1. Load trained model weights
  2. Compile ONLY QuantVGG11Patch (no max aggregation) to FHE
  3. Verify accuracy using FHE simulation (fhe="simulate")
  4. Sweep rounding_threshold_bits to find speed/accuracy tradeoff

The max-logit aggregation across 49 patches happens in plain Python,
completely outside the FHE circuit.

Usage:
  python fhe_convert.py --data-dir ./data --model-path quant_vgg11_patch.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from concrete.ml.torch.compile import compile_brevitas_qat_model

from model import QuantVGG11Patch, PATCH_SIZE, GRID_SIZE, NUM_PATCHES, NUM_CLASSES
from train import build_datasets


def split_image_to_patches_numpy(image_tensor):
    """
    Split a single [3, 224, 224] tensor into [49, 3, 32, 32] numpy array.
    Done in plain Python -- never part of FHE.
    """
    C, H, W = image_tensor.shape
    patches = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            patch = image_tensor[
                :,
                row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                col * PATCH_SIZE : (col + 1) * PATCH_SIZE,
            ]
            patches.append(patch.numpy())
    return np.array(patches)  # [49, 3, 32, 32]


def build_calibration_data(dataset, num_samples=100):
    """
    Grab random patches from the dataset for calibration.
    Returns numpy array [num_samples, 3, 32, 32].
    """
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    patches_list = []
    for idx in indices:
        image, _ = dataset[idx]
        # grab one random patch from this image
        all_patches = split_image_to_patches_numpy(image)
        patch_idx = rng.integers(0, NUM_PATCHES)
        patches_list.append(all_patches[patch_idx])

    return np.array(patches_list, dtype=np.float32)  # [num_samples, 3, 32, 32]


def evaluate_fhe_simulation(quantized_module, dataset, max_images=None):
    """
    Run FHE simulation on val set.
    For each image: split into 49 patches, run each through the compiled module
    with fhe="simulate", take max logit per class (in plain Python), compare to label.
    """
    correct = 0
    total = 0
    num_images = len(dataset) if max_images is None else min(max_images, len(dataset))

    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES

    for i in range(num_images):
        image, label = dataset[i]
        patches = split_image_to_patches_numpy(image)  # [49, 3, 32, 32]

        # run each patch through the FHE-simulated model
        all_logits = []
        for p in range(NUM_PATCHES):
            patch = patches[p : p + 1]  # [1, 3, 32, 32]
            logits = quantized_module.forward(patch, fhe="simulate")  # [1, 7]
            all_logits.append(logits)

        # aggregate: max logit per class across all patches (plain Python, outside FHE)
        all_logits = np.concatenate(all_logits, axis=0)  # [49, 7]
        max_logits = np.max(all_logits, axis=0)  # [7]
        pred = np.argmax(max_logits)

        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        per_class_total[label] += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  evaluated {i + 1}/{num_images} images, running acc: {correct / total:.4f}")

    accuracy = correct / total

    print(f"\nFHE simulation accuracy: {accuracy:.4f} ({correct}/{total})")
    print("per-class recall:")
    for c in range(NUM_CLASSES):
        if per_class_total[c] > 0:
            recall = per_class_correct[c] / per_class_total[c]
            print(f"  class {c}: {recall:.3f} ({per_class_correct[c]}/{per_class_total[c]})")

    return accuracy


def compile_and_evaluate(model, calibration_data, val_dataset, rounding_n_bits, max_eval_images=None):
    """
    Compile model with given rounding_threshold_bits, then evaluate via FHE simulation.
    """
    print(f"\n{'='*60}")
    print(f"compiling with rounding_threshold_bits n_bits={rounding_n_bits}")
    print(f"{'='*60}")

    quantized_module = compile_brevitas_qat_model(
        model,
        torch.from_numpy(calibration_data),
        rounding_threshold_bits={"n_bits": rounding_n_bits, "method": "approximate"},
    )

    print("compilation successful")
    accuracy = evaluate_fhe_simulation(quantized_module, val_dataset, max_images=max_eval_images)
    return accuracy, quantized_module


def main():
    parser = argparse.ArgumentParser(description="compile QuantVGG11Patch to FHE circuit")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-path", type=str, default="quant_vgg11_patch.pth")
    parser.add_argument("--num-calibration", type=int, default=100, help="num patches for calibration")
    parser.add_argument("--max-eval-images", type=int, default=200, help="max val images for simulation eval (None=all)")
    parser.add_argument("--sweep-rounding", action="store_true", help="sweep rounding n_bits from 8 down to 4")
    args = parser.parse_args()

    # load trained model (on CPU -- FHE compilation is CPU-only)
    model = QuantVGG11Patch()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    print("loaded trained model")

    # build val dataset for evaluation
    _, val_ds, _ = build_datasets(args.data_dir)

    # build calibration data (random single patches)
    calibration_data = build_calibration_data(val_ds, num_samples=args.num_calibration)
    print(f"calibration data shape: {calibration_data.shape}")

    if args.sweep_rounding:
        # sweep rounding_threshold_bits to find speed/accuracy tradeoff
        results = {}
        for n_bits in [8, 7, 6, 5, 4]:
            try:
                acc, _ = compile_and_evaluate(
                    model, calibration_data, val_ds, n_bits,
                    max_eval_images=args.max_eval_images,
                )
                results[n_bits] = acc
            except Exception as e:
                print(f"  n_bits={n_bits} failed: {e}")
                results[n_bits] = None

        print(f"\n{'='*60}")
        print("rounding sweep results:")
        print(f"{'='*60}")
        for n_bits, acc in sorted(results.items(), reverse=True):
            status = f"{acc:.4f}" if acc is not None else "FAILED"
            print(f"  n_bits={n_bits}: accuracy={status}")

        # pick the lowest n_bits with acceptable accuracy
        best_n_bits = None
        for n_bits in sorted(results.keys()):
            if results[n_bits] is not None:
                best_n_bits = n_bits
        if best_n_bits:
            print(f"\nlowest working n_bits: {best_n_bits} (accuracy={results[best_n_bits]:.4f})")
    else:
        # single compilation with default n_bits=6
        acc, quantized_module = compile_and_evaluate(
            model, calibration_data, val_ds, rounding_n_bits=6,
            max_eval_images=args.max_eval_images,
        )
        print(f"\nfinal FHE simulation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
