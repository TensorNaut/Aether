"""
Patch Extractor for OHRC Images
================================
Extracts 64x64 patches from full OHRC strips and classifies them
as PSR / SUNLIT / MIXED based on pixel statistics.

Patches are saved as .npy files organized by type, ready for
Kaggle upload and Zero-DCE training.

Usage:
    python scripts/extract_patches.py
"""

import sys
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_io.pds4_reader import load_ohrc_image, image_statistics


def extract_patches_from_image(
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    dark_threshold: float = 15.0 / 255.0,
    bright_threshold: float = 50.0 / 255.0,
    min_std: float = 0.005,
) -> dict:
    """Extract and classify patches from a single OHRC image.

    Args:
        image: float32 array [0, 1], shape (H, W)
        patch_size: Size of square patches
        stride: Stride between patches
        dark_threshold: Pixels below this are dark
        bright_threshold: Minimum mean for sunlit
        min_std: Minimum std to keep (rejects flat/dead patches)

    Returns:
        Dict with keys 'psr', 'sunlit', 'mixed', each containing
        list of float32 patches of shape (patch_size, patch_size)
    """
    H, W = image.shape
    patches = {'psr': [], 'sunlit': [], 'mixed': []}

    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1

    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride
            patch = image[y:y + patch_size, x:x + patch_size]

            # Skip dead/flat patches
            if np.std(patch) < min_std:
                continue

            # Classify
            mean_val = np.mean(patch)
            pct_dark = np.mean(patch < dark_threshold) * 100

            if pct_dark > 85:
                # Mostly dark — PSR patch
                # But skip if completely black (no signal)
                if np.max(patch) > 0.01:
                    patches['psr'].append(patch)
            elif pct_dark < 20 and mean_val > bright_threshold:
                # Mostly bright — sunlit patch
                patches['sunlit'].append(patch)
            elif 20 <= pct_dark <= 85:
                # Mix of dark and light — boundary region
                patches['mixed'].append(patch)

    return patches


def save_patches(
    patches: dict,
    output_dir: str,
    image_name: str,
    max_per_type: int = 5000,
):
    """Save extracted patches as .npy files.

    Saves individual patches and a consolidated array per type.
    """
    output_dir = Path(output_dir)

    for ptype, patch_list in patches.items():
        if not patch_list:
            continue

        type_dir = output_dir / ptype
        type_dir.mkdir(parents=True, exist_ok=True)

        # Limit patches per type to avoid huge files
        if len(patch_list) > max_per_type:
            # Random subsample
            rng = np.random.default_rng(42)
            indices = rng.choice(len(patch_list), max_per_type, replace=False)
            patch_list = [patch_list[i] for i in indices]

        # Save as single consolidated array
        arr = np.stack(patch_list, axis=0)  # (N, 64, 64) float32
        save_path = type_dir / f"{image_name}.npy"
        np.save(str(save_path), arr)

        print(f"  {ptype}: {len(patch_list)} patches -> {save_path.name} "
              f"({arr.nbytes / (1024*1024):.1f} MB)")


def create_training_splits(
    patches_dir: str,
    train_ratio: float = 0.85,
    seed: int = 42,
):
    """Create train/val splits from extracted patches.

    Loads all .npy files per type, shuffles, splits, and saves
    consolidated train.npy and val.npy files.
    """
    patches_dir = Path(patches_dir)
    rng = np.random.default_rng(seed)

    for ptype in ['psr', 'sunlit', 'mixed']:
        type_dir = patches_dir / ptype
        if not type_dir.exists():
            continue

        # Load all .npy files for this type
        all_patches = []
        for npy_file in sorted(type_dir.glob('*.npy')):
            if npy_file.name in ('train.npy', 'val.npy'):
                continue
            arr = np.load(str(npy_file))
            all_patches.append(arr)
            print(f"  Loaded {npy_file.name}: {arr.shape}")

        if not all_patches:
            continue

        combined = np.concatenate(all_patches, axis=0)
        rng.shuffle(combined)

        n_train = int(len(combined) * train_ratio)
        train = combined[:n_train]
        val = combined[n_train:]

        np.save(str(type_dir / 'train.npy'), train)
        np.save(str(type_dir / 'val.npy'), val)

        print(f"  {ptype}: {len(train)} train + {len(val)} val "
              f"(total {len(combined)})")


def main():
    parser = argparse.ArgumentParser(
        description='Extract patches from OHRC images for training'
    )
    parser.add_argument(
        '--data-dir',
        default=r'D:\Projects and Coding\Version Control Systems\Aether_Data',
        help='Directory containing extracted OHRC images'
    )
    parser.add_argument(
        '--output-dir',
        default=r'D:\Projects and Coding\Version Control Systems\Aether_Data\patches',
        help='Where to save extracted patches'
    )
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=48,
                        help='Stride between patches (smaller = more patches, more overlap)')
    parser.add_argument('--max-per-type', type=int, default=5000,
                        help='Max patches per type per image')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all OHRC images
    images = {
        'shackleton_01': data_dir / 'batch_01' / 'ch2_ohr_ncp_20251007T0659402125_d_img_d18' / 'data' / 'calibrated' / '20251007' / 'ch2_ohr_ncp_20251007T0659402125_d_img_d18.xml',
        'shackleton_02': data_dir / 'batch_01' / 'ch2_ohr_ncp_20251105T0652329256_d_img_d18' / 'data' / 'calibrated' / '20251105' / 'ch2_ohr_ncp_20251105T0652329256_d_img_d18.xml',
        'cabeus_01': data_dir / 'ch2_ohr_ncp_20250310T0833447498_d_img_d18' / 'data' / 'calibrated' / '20250310' / 'ch2_ohr_ncp_20250310T0833447498_d_img_d18.xml',
    }

    sep = '=' * 60
    print(f"\n{sep}")
    print(f" AETHER - OHRC Patch Extractor")
    print(f" Patch size: {args.patch_size}, Stride: {args.stride}")
    print(f" Output: {output_dir}")
    print(f"{sep}")

    for name, label_path in images.items():
        print(f"\n--- {name} ---")

        if not label_path.exists():
            print(f"  SKIP: {label_path} not found")
            continue

        # Load image
        print(f"  Loading image...")
        img = load_ohrc_image(str(label_path))
        stats = image_statistics(img)
        print(f"  Shape: {stats['shape']}, Mean: {stats['mean']:.1f}/255, "
              f"Dark: {stats['pct_dark']:.1f}%")

        # Extract patches
        print(f"  Extracting patches...")
        patches = extract_patches_from_image(
            img,
            patch_size=args.patch_size,
            stride=args.stride,
        )

        total = sum(len(v) for v in patches.values())
        print(f"  Found: {len(patches['psr'])} PSR, "
              f"{len(patches['sunlit'])} sunlit, "
              f"{len(patches['mixed'])} mixed "
              f"({total} total)")

        # Save
        save_patches(patches, str(output_dir), name, args.max_per_type)

        # Free memory
        del img
        import gc
        gc.collect()

    # Create train/val splits
    print(f"\n--- Creating train/val splits ---")
    create_training_splits(str(output_dir))

    # Print summary
    print(f"\n{sep}")
    print(f" DONE! Patches saved to: {output_dir}")
    print(f"{sep}")

    # Print sizes
    total_size = 0
    for npy in output_dir.rglob('*.npy'):
        size = os.path.getsize(npy) / (1024 * 1024)
        total_size += size

    print(f"\n  Total patches size: {total_size:.1f} MB")
    print(f"  Ready for Kaggle upload!")


if __name__ == '__main__':
    main()
