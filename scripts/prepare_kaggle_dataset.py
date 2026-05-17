"""
Prepare Kaggle Dataset
======================
Organizes all OHRC data into a clean folder structure for
uploading as a Kaggle dataset.

Structure:
    kaggle_dataset/
    ├── raw/
    │   ├── shackleton_01/
    │   │   ├── image.img        (raw binary)
    │   │   ├── label.xml        (PDS4 metadata)
    │   │   ├── geometry.csv     (lat/lon per pixel)
    │   │   └── browse.png       (preview)
    │   ├── shackleton_02/
    │   └── cabeus_01/
    ├── patches/
    │   ├── psr/
    │   │   ├── train.npy
    │   │   └── val.npy
    │   ├── sunlit/
    │   │   ├── train.npy
    │   │   └── val.npy
    │   └── mixed/
    │       ├── train.npy
    │       └── val.npy
    ├── metadata.json
    └── README.md

Usage:
    python scripts/prepare_kaggle_dataset.py
"""

import sys
import os
import json
import shutil
import glob
from pathlib import Path
from datetime import datetime


def main():
    data_dir = Path(r'D:\Projects and Coding\Version Control Systems\Aether_Data')
    kaggle_dir = data_dir / 'kaggle_dataset'
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    # Image mappings: friendly name -> source directory
    images = {
        'shackleton_01': {
            'source': data_dir / 'batch_01' / 'ch2_ohr_ncp_20251007T0659402125_d_img_d18',
            'date': '20251007',
            'product_id': 'ch2_ohr_ncp_20251007T0659402125_d_img_d18',
            'crater': 'Shackleton',
            'lat_range': '-89.72 to -89.08',
            'classification': 'PSR',
        },
        'shackleton_02': {
            'source': data_dir / 'batch_01' / 'ch2_ohr_ncp_20251105T0652329256_d_img_d18',
            'date': '20251105',
            'product_id': 'ch2_ohr_ncp_20251105T0652329256_d_img_d18',
            'crater': 'Shackleton',
            'lat_range': '-89.90 to -89.11',
            'classification': 'PSR',
        },
        'cabeus_01': {
            'source': data_dir / 'ch2_ohr_ncp_20250310T0833447498_d_img_d18',
            'date': '20250310',
            'product_id': 'ch2_ohr_ncp_20250310T0833447498_d_img_d18',
            'crater': 'Cabeus',
            'lat_range': '-86.37 to -85.55',
            'classification': 'MIXED',
        },
    }

    sep = '=' * 60
    print(f"\n{sep}")
    print(f" Preparing Kaggle Dataset")
    print(f"{sep}")

    # --- Copy raw images ---
    raw_dir = kaggle_dir / 'raw'
    for name, info in images.items():
        src = info['source']
        dst = raw_dir / name
        dst.mkdir(parents=True, exist_ok=True)

        date = info['date']
        pid = info['product_id']

        # Copy .img file
        img_files = list(src.rglob('*_d_img_*.img'))
        if img_files:
            target = dst / 'image.img'
            if not target.exists():
                print(f"  Copying {name}/image.img ({img_files[0].stat().st_size/(1024**2):.0f} MB)...")
                shutil.copy2(str(img_files[0]), str(target))
            else:
                print(f"  {name}/image.img already exists, skipping")

        # Copy .xml label
        xml_files = list(src.rglob('*_d_img_*.xml'))
        if xml_files:
            shutil.copy2(str(xml_files[0]), str(dst / 'label.xml'))

        # Copy geometry CSV
        csv_files = list(src.rglob('*.csv'))
        if csv_files:
            shutil.copy2(str(csv_files[0]), str(dst / 'geometry.csv'))

        # Copy browse PNG
        png_files = list(src.rglob('*_brw_*.png'))
        if png_files:
            shutil.copy2(str(png_files[0]), str(dst / 'browse.png'))

        print(f"  {name}: done")

    # --- Copy patches ---
    patches_src = data_dir / 'patches'
    patches_dst = kaggle_dir / 'patches'
    if patches_src.exists():
        print(f"\n  Copying patches...")
        if patches_dst.exists():
            shutil.rmtree(str(patches_dst))
        shutil.copytree(str(patches_src), str(patches_dst))
        print(f"  Patches copied")
    else:
        print(f"\n  WARNING: No patches found at {patches_src}")
        print(f"  Run extract_patches.py first!")

    # --- Create metadata.json ---
    metadata = {
        'name': 'chandrayaan2-ohrc-psr-dataset',
        'title': 'Chandrayaan-2 OHRC - Lunar Permanently Shadowed Regions',
        'description': (
            'Calibrated OHRC (Orbiter High Resolution Camera) images from '
            'ISRO Chandrayaan-2 mission, targeting Permanently Shadowed '
            'Regions (PSRs) near the lunar south pole. Includes images from '
            'Shackleton and Cabeus craters at 0.25 m/pixel resolution.'
        ),
        'mission': 'Chandrayaan-2',
        'instrument': 'OHRC (Orbiter High Resolution Camera)',
        'resolution': '0.25 m/pixel',
        'data_format': 'PDS4 (uint8 binary + XML label)',
        'source': 'ISRO PRADAN (pradan.issdc.gov.in)',
        'created': datetime.now().isoformat(),
        'images': {},
    }

    for name, info in images.items():
        metadata['images'][name] = {
            'crater': info['crater'],
            'date': info['date'],
            'product_id': info['product_id'],
            'lat_range': info['lat_range'],
            'classification': info['classification'],
            'dimensions': '93692x12000' if 'cabeus' in name else '101075x12000',
        }

    with open(str(kaggle_dir / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # --- Create dataset README ---
    readme = """# Chandrayaan-2 OHRC - Lunar PSR Dataset

## Overview
Calibrated images from ISRO's Chandrayaan-2 Orbiter High Resolution Camera (OHRC),
targeting Permanently Shadowed Regions (PSRs) near the lunar south pole.

## Images

| Name | Crater | Latitude | Type | Dimensions |
|------|--------|----------|------|------------|
| shackleton_01 | Shackleton | 89.1-89.7°S | PSR (91% dark) | 101075 × 12000 |
| shackleton_02 | Shackleton | 89.1-89.9°S | PSR (81% dark) | 101075 × 12000 |
| cabeus_01 | Cabeus | 85.6-86.4°S | MIXED (68% dark) | 93692 × 12000 |

## Data Format
- `raw/*/image.img` — Raw uint8 binary (no header), row-major
- `raw/*/label.xml` — PDS4 XML metadata (dimensions, observation time)
- `raw/*/geometry.csv` — Pixel-level lat/lon coordinates
- `raw/*/browse.png` — Low-resolution preview

## Loading
```python
import numpy as np

# Load raw image
lines, samples = 101075, 12000  # from label.xml
img = np.fromfile('raw/shackleton_01/image.img', dtype=np.uint8)
img = img.reshape(lines, samples).astype(np.float32) / 255.0
```

## Patches (Pre-extracted)
- `patches/psr/train.npy` — Dark PSR patches (64×64, float32)
- `patches/sunlit/train.npy` — Well-illuminated patches
- `patches/mixed/train.npy` — Shadow boundary patches

## Source
Downloaded from [ISRO PRADAN](https://pradan.issdc.gov.in) via
[CH2 MapBrowse](https://chmapbrowse.issdc.gov.in).

## Citation
ISRO/ISSDC, Chandrayaan-2 OHRC Calibrated Data Products, 2024-2025.

## License
ISRO Open Data Policy — free for research and educational use.
"""

    with open(str(kaggle_dir / 'README.md'), 'w') as f:
        f.write(readme)

    # --- Print summary ---
    print(f"\n{sep}")
    print(f" Dataset prepared at: {kaggle_dir}")
    print(f"{sep}")

    total_size = 0
    for root, dirs, files in os.walk(str(kaggle_dir)):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)

    print(f"\n  Total size: {total_size/(1024**3):.2f} GB")
    print(f"\n  To upload to Kaggle:")
    print(f"  1. Go to kaggle.com/datasets -> New Dataset")
    print(f"  2. Upload the folder: {kaggle_dir}")
    print(f"  3. Or use Kaggle API:")
    print(f"     kaggle datasets create -p \"{kaggle_dir}\"")


if __name__ == '__main__':
    main()
