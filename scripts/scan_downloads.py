"""
OHRC Image Scanner
==================
Scans a directory of downloaded OHRC images (extracted from PRADAN tars/zips)
and classifies them as PSR / MIXED / SUNLIT based on pixel statistics.

This solves the problem of PRADAN not showing coordinates —
we just download a batch and let the code sort them out.

Usage:
    python scripts/scan_downloads.py D:\Aether_Data\shackleton\batch_01
"""

import sys
import os
import argparse
import zipfile
import tarfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data_io.pds4_reader import (
    load_ohrc_image,
    load_ohrc_metadata,
    image_statistics,
    quick_preview,
    find_label_files,
)


def extract_tar(tar_path: str, output_dir: str) -> list:
    """Extract a tar file and return list of extracted directories."""
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    print(f"\nExtracting tar: {tar_path.name}")

    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(output_dir)
        for member in tar.getnames():
            full_path = output_dir / member
            if full_path.exists():
                extracted.append(str(full_path))

    print(f"  Extracted {len(extracted)} items to {output_dir}")
    return extracted


def extract_zip(zip_path: str, output_dir: str = None) -> str:
    """Extract a zip file, return the extraction directory."""
    zip_path = Path(zip_path)
    if output_dir is None:
        output_dir = zip_path.parent / zip_path.stem
    else:
        output_dir = Path(output_dir) / zip_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    return str(output_dir)


def scan_browse_images(directory: str) -> list:
    """Find all browse PNG images for quick visual scanning."""
    directory = Path(directory)
    browse_pngs = []

    for png in directory.rglob('*brw*.png'):
        browse_pngs.append(str(png))
    for png in directory.rglob('browse/*.png'):
        if str(png) not in browse_pngs:
            browse_pngs.append(str(png))

    return sorted(browse_pngs)


def scan_directory(directory: str, extract_zips: bool = True,
                   save_previews: bool = True) -> list:
    """Scan a directory of OHRC downloads and classify each image.

    Args:
        directory: Path to directory containing tars/zips/extracted OHRC data
        extract_zips: Whether to auto-extract any found ZIP files
        save_previews: Whether to save gamma-corrected preview PNGs

    Returns:
        List of dicts with classification info for each image.
    """
    directory = Path(directory)
    results = []

    # Step 1: Extract any tar files
    for tar_file in directory.glob('*.tar*'):
        extract_dir = directory / tar_file.stem
        if not extract_dir.exists():
            extract_tar(str(tar_file), str(extract_dir))

    # Step 2: Extract any zip files
    if extract_zips:
        for zip_file in directory.rglob('*.zip'):
            extract_dir = zip_file.parent / zip_file.stem
            if not extract_dir.exists():
                print(f"  Extracting ZIP: {zip_file.name}")
                try:
                    extract_zip(str(zip_file), str(zip_file.parent))
                except Exception as e:
                    print(f"    Failed: {e}")

    # Step 3: Find all PDS4 label files
    label_files = find_label_files(str(directory))

    if not label_files:
        print(f"\nNo PDS4 label files found in {directory}")
        print("Looking for browse PNGs instead...")
        browse_pngs = scan_browse_images(str(directory))
        if browse_pngs:
            print(f"Found {len(browse_pngs)} browse PNGs:")
            for png in browse_pngs:
                print(f"  {png}")
        return []

    print(f"\nFound {len(label_files)} OHRC label files")
    print(f"{'='*80}")

    # Step 4: Load and classify each image
    preview_dir = directory / 'previews'
    if save_previews:
        preview_dir.mkdir(exist_ok=True)

    for i, label_path in enumerate(label_files):
        label_name = Path(label_path).stem
        print(f"\n[{i+1}/{len(label_files)}] {label_name}")

        try:
            # Load image
            img = load_ohrc_image(label_path)
            meta = load_ohrc_metadata(label_path)
            stats = image_statistics(img)

            result = {
                'index': i + 1,
                'label_path': label_path,
                'filename': label_name,
                'shape': stats['shape'],
                'classification': stats['classification'],
                'mean_pixel': stats['mean'],
                'pct_dark': stats['pct_dark'],
                'pct_bright': stats['pct_bright'],
                'estimated_snr': stats['estimated_snr'],
                'start_time': meta.get('start_time', 'unknown'),
            }
            results.append(result)

            # Print summary
            cls_emoji = {
                'PSR': '🌑',
                'MIXED': '🌓',
                'SUNLIT': '☀️',
            }
            cls_key = stats['classification'].split('(')[0].strip()
            emoji = cls_emoji.get(cls_key, '❓')

            print(f"  {emoji} {stats['classification']}")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean pixel: {stats['mean']:.1f}/255")
            print(f"  Dark pixels: {stats['pct_dark']:.1f}%")
            print(f"  Bright pixels: {stats['pct_bright']:.1f}%")
            print(f"  Est. SNR: {stats['estimated_snr']:.3f}")

            # Save preview
            if save_previews:
                preview = quick_preview(img, gamma=0.3)
                preview_path = preview_dir / f"{label_name}_preview.png"
                try:
                    from PIL import Image
                    Image.fromarray(preview).save(str(preview_path))
                    print(f"  Preview: {preview_path}")
                except ImportError:
                    import cv2
                    cv2.imwrite(str(preview_path), preview)
                    print(f"  Preview: {preview_path}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'index': i + 1,
                'label_path': label_path,
                'filename': label_name,
                'classification': f'ERROR: {e}',
                'error': str(e),
            })

    return results


def print_summary(results: list):
    """Print a summary table of scan results."""
    print(f"\n\n{'='*80}")
    print(f" SCAN SUMMARY")
    print(f"{'='*80}\n")

    psr = [r for r in results if 'PSR' in r.get('classification', '')]
    mixed = [r for r in results if 'MIXED' in r.get('classification', '')]
    sunlit = [r for r in results if 'SUNLIT' in r.get('classification', '')]
    errors = [r for r in results if 'ERROR' in r.get('classification', '')]

    print(f"  🌑 PSR (dark — enhancement targets):  {len(psr)}")
    print(f"  🌓 MIXED (partial shadow):             {len(mixed)}")
    print(f"  ☀️  SUNLIT (bright — training data):    {len(sunlit)}")
    print(f"  ❌ ERRORS:                              {len(errors)}")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL:                                 {len(results)}")

    if psr:
        print(f"\n  📌 PSR images (USE THESE for enhancement):")
        for r in psr:
            print(f"     → {r['filename']}")
            print(f"       Dark pixels: {r.get('pct_dark', '?'):.1f}%, "
                  f"Mean: {r.get('mean_pixel', '?'):.1f}/255")

    if sunlit:
        print(f"\n  📌 SUNLIT images (USE THESE for training data):")
        for r in sunlit[:5]:  # Show max 5
            print(f"     → {r['filename']}")

    if not psr:
        print(f"\n  ⚠️  No PSR images found in this batch!")
        print(f"  Try downloading images from a different region or batch.")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scan downloaded OHRC images and classify PSR vs Sunlit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a directory of extracted tar/zip files:
  python scripts/scan_downloads.py D:\\Aether_Data\\shackleton

  # Scan without extracting ZIPs (if already extracted):
  python scripts/scan_downloads.py data/ohrc/psr --no-extract

  # Scan without saving previews (faster):
  python scripts/scan_downloads.py data/ohrc --no-previews
        """
    )
    parser.add_argument('directory', help='Directory containing OHRC downloads')
    parser.add_argument('--no-extract', action='store_true',
                        help='Skip ZIP extraction')
    parser.add_argument('--no-previews', action='store_true',
                        help='Skip saving preview PNGs')

    args = parser.parse_args()

    if not Path(args.directory).exists():
        print(f"Error: Directory does not exist: {args.directory}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f" AETHER — OHRC Download Scanner")
    print(f" Scanning: {args.directory}")
    print(f"{'='*80}")

    results = scan_directory(
        args.directory,
        extract_zips=not args.no_extract,
        save_previews=not args.no_previews,
    )

    if results:
        print_summary(results)
