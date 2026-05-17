"""
PDS4 OHRC Image Reader
======================
Reads Chandrayaan-2 OHRC images from their PDS4 (.img + .xml) format.

OHRC produces panchromatic images at 0.25 m/pixel resolution.
Data type: uint8 (0-255), up to 54860 x 12000 pixels per strip.

Reference: SAC/SIPG (2019), Chandrayaan-2 OHRC PDS4 Data Products User Guide.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import xml.etree.ElementTree as ET


def load_ohrc_image(label_path: str) -> np.ndarray:
    """Load an OHRC PDS4 image from its XML label file.

    The XML label describes the binary .img file layout.
    We try pds4_tools first; if unavailable, fall back to manual parsing.

    Args:
        label_path: Path to the .xml PDS4 label file.

    Returns:
        float32 numpy array normalised to [0, 1], shape (H, W).
    """
    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Try pds4_tools first (cleanest approach)
    try:
        import pds4_tools
        structures = pds4_tools.read(str(label_path), quiet=True)
        img_struct = structures[0]
        data = img_struct.data
        img = data.astype(np.float32) / 255.0
        return img
    except ImportError:
        pass  # Fall through to manual parsing
    except Exception as e:
        print(f"pds4_tools failed ({e}), trying manual parse...")

    # Fallback: manual PDS4 XML + binary parsing
    return _manual_load(label_path)


def _manual_load(label_path: Path) -> np.ndarray:
    """Manually parse PDS4 XML label and read raw binary image.

    This is the fallback when pds4_tools is not available.
    Parses the XML to find image dimensions and data file path,
    then reads the raw binary as uint8.
    """
    tree = ET.parse(label_path)
    root = tree.getroot()

    # PDS4 uses XML namespaces — we need to handle them
    # Common namespace for PDS4 core
    ns = {}
    for key, val in root.attrib.items():
        if 'xmlns' in key:
            prefix = key.split('}')[0].split('{')[-1] if '{' in key else ''
            if not prefix:
                prefix = key.replace('xmlns:', '') if ':' in key else ''
            ns[prefix] = val

    # Try to find namespaces from the root tag
    root_tag = root.tag
    default_ns = ''
    if '{' in root_tag:
        default_ns = root_tag.split('}')[0] + '}'

    # Find image dimensions — search for Array_2D or Array_2D_Image
    lines, samples = None, None
    data_file = None

    # Search all elements for axis information
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == 'lines':
            lines = int(elem.text.strip())
        elif tag == 'samples':
            samples = int(elem.text.strip())
        elif tag == 'file_name':
            data_file = elem.text.strip()
        elif tag == 'axes' and lines is None:
            pass  # Just a marker
        elif tag == 'axis_name' and elem.text:
            pass  # We handle via sequence_number
        elif tag == 'elements':
            # Elements in an axis — could be lines or samples
            parent_tag = ''
            for parent in root.iter():
                for child in parent:
                    if child is elem:
                        parent_tag = parent.tag.split('}')[-1] if '}' in parent.tag else parent.tag
                        break

    # If we couldn't find dimensions from specific tags, try Axis_Array
    if lines is None or samples is None:
        axes = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'Axis_Array':
                axis_name = None
                elements = None
                for child in elem:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag == 'axis_name':
                        axis_name = child.text.strip()
                    elif child_tag == 'elements':
                        elements = int(child.text.strip())
                if axis_name and elements:
                    axes.append((axis_name, elements))

        for name, count in axes:
            if name.lower() in ('line', 'lines'):
                lines = count
            elif name.lower() in ('sample', 'samples'):
                samples = count

    if lines is None or samples is None:
        raise ValueError(
            f"Could not determine image dimensions from {label_path}. "
            f"Found lines={lines}, samples={samples}"
        )

    # Find the data file
    if data_file is None:
        # Default: same name as label but with .img extension
        data_file = label_path.stem + '.img'

    img_path = label_path.parent / data_file
    if not img_path.exists():
        # Try case-insensitive search
        for f in label_path.parent.iterdir():
            if f.name.lower() == data_file.lower():
                img_path = f
                break

    if not img_path.exists():
        raise FileNotFoundError(
            f"Image data file not found: {img_path}\n"
            f"Expected alongside label: {label_path}"
        )

    # Read raw binary — OHRC is uint8
    raw = np.fromfile(str(img_path), dtype=np.uint8)

    # Reshape to image dimensions
    expected_size = lines * samples
    if raw.size < expected_size:
        raise ValueError(
            f"Data file too small: {raw.size} bytes, "
            f"expected {expected_size} ({lines} x {samples})"
        )

    # Take only the expected number of bytes (skip any header/padding)
    img = raw[:expected_size].reshape(lines, samples)

    return img.astype(np.float32) / 255.0


def load_ohrc_metadata(label_path: str) -> dict:
    """Extract key metadata from OHRC PDS4 XML label.

    Returns dict with observation time, image dimensions,
    and any available coordinate information.
    """
    label_path = Path(label_path)
    tree = ET.parse(label_path)
    root = tree.getroot()

    metadata = {
        'label_path': str(label_path),
        'image_path': None,
        'lines': None,
        'samples': None,
        'start_time': None,
        'stop_time': None,
        'exposure_duration': None,
    }

    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        text = elem.text.strip() if elem.text else None

        if tag == 'lines' and text:
            metadata['lines'] = int(text)
        elif tag == 'samples' and text:
            metadata['samples'] = int(text)
        elif tag == 'start_date_time' and text:
            metadata['start_time'] = text
        elif tag == 'stop_date_time' and text:
            metadata['stop_time'] = text
        elif tag == 'file_name' and text:
            metadata['image_path'] = str(label_path.parent / text)
        elif tag == 'exposure_duration' and text:
            metadata['exposure_duration'] = text

    return metadata


def quick_preview(image: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    """Generate a gamma-corrected preview of an OHRC image.

    PSR images are mostly dark (pixel values 1-10/255).
    Gamma correction with γ < 1 brightens the dark regions
    so we can see what's there.

    Args:
        image: float32 array [0, 1], shape (H, W)
        gamma: gamma correction value (0.3 = aggressive brightening)

    Returns:
        uint8 array [0, 255], shape (H, W) — ready for display/save
    """
    # Clip and apply gamma
    preview = np.clip(image, 0, 1)
    preview = np.power(preview, gamma)
    return (preview * 255).astype(np.uint8)


def find_label_files(directory: str) -> list:
    """Recursively find all PDS4 XML label files in a directory.

    Looks for .xml files that are PDS4 labels (not other XML files).
    """
    directory = Path(directory)
    label_files = []

    for xml_file in directory.rglob('*.xml'):
        # Quick check: is this a PDS4 label?
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                header = f.read(500)
            if 'Product_Observational' in header or 'pds4' in header.lower():
                label_files.append(str(xml_file))
        except (UnicodeDecodeError, IOError):
            continue

    return sorted(label_files)


def image_statistics(image: np.ndarray) -> dict:
    """Compute basic statistics for an OHRC image.

    Useful for quickly identifying PSR vs sunlit images:
    - PSR images: mean < 0.04, most pixels < 15/255
    - Sunlit images: mean > 0.2, wide pixel distribution
    """
    # Convert back to 0-255 range for interpretability
    img_255 = image * 255.0

    stats = {
        'shape': image.shape,
        'mean': float(np.mean(img_255)),
        'std': float(np.std(img_255)),
        'min': float(np.min(img_255)),
        'max': float(np.max(img_255)),
        'median': float(np.median(img_255)),
        'pct_dark': float(np.mean(img_255 < 15) * 100),  # % pixels below 15/255
        'pct_bright': float(np.mean(img_255 > 100) * 100),  # % pixels above 100/255
        'estimated_snr': float(np.mean(img_255) / (np.std(img_255) + 1e-8)),
    }

    # Classification heuristic
    if stats['pct_dark'] > 70:
        stats['classification'] = 'PSR (mostly dark — enhancement target)'
    elif stats['pct_dark'] > 30:
        stats['classification'] = 'MIXED (partial shadow — usable)'
    else:
        stats['classification'] = 'SUNLIT (bright — training data source)'

    return stats


if __name__ == '__main__':
    """Quick test: load and inspect an OHRC image."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Inspect an OHRC PDS4 image')
    parser.add_argument('label', help='Path to PDS4 .xml label file')
    parser.add_argument('--save-preview', help='Save gamma-corrected preview PNG')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" AETHER — OHRC Image Inspector")
    print(f"{'='*60}\n")

    # Load metadata
    meta = load_ohrc_metadata(args.label)
    print("Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Load image
    print(f"\nLoading image...")
    img = load_ohrc_image(args.label)

    # Stats
    stats = image_statistics(img)
    print(f"\nImage Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Save preview
    if args.save_preview:
        from PIL import Image
        preview = quick_preview(img, gamma=0.3)
        Image.fromarray(preview).save(args.save_preview)
        print(f"\nPreview saved to: {args.save_preview}")

    print(f"\n{'='*60}")
