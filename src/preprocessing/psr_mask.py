"""
PSR Mask Generation
===================
Identifies Permanently Shadowed Region pixels in OHRC images.

Since we don't have LOLA DEM aligned to every image, we use a
photometric approach: pixels below a brightness threshold are
classified as PSR. This works well because OHRC PSR floors
typically have pixel values of 1-10/255 while sunlit regions
are 100-255/255.

Reference: plan.pdf, Algorithm 1 — PSR Mask Generation
"""

import numpy as np
from typing import Tuple


def generate_psr_mask(
    image: np.ndarray,
    dark_threshold: float = 15.0 / 255.0,
    closing_kernel_size: int = 5,
    min_region_size: int = 100,
) -> np.ndarray:
    """Generate a binary PSR mask from an OHRC image.

    Uses photometric thresholding: pixels darker than threshold
    are classified as PSR. Morphological closing fills small gaps.

    Args:
        image: float32 array [0, 1], shape (H, W)
        dark_threshold: Pixels below this are PSR (default 15/255 ≈ 0.059)
        closing_kernel_size: Size of morphological closing kernel
        min_region_size: Minimum connected component size to keep

    Returns:
        Binary mask (H, W), uint8: 1 = PSR, 0 = illuminated
    """
    # Photometric thresholding
    mask = (image < dark_threshold).astype(np.uint8)

    # Morphological closing to fill small gaps
    try:
        import cv2
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_kernel_size, closing_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove small connected components (noise)
        if min_region_size > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < min_region_size:
                    mask[labels == label_id] = 0
    except ImportError:
        # Fallback without OpenCV
        from scipy import ndimage
        struct = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_closing(
            mask, structure=struct, iterations=closing_kernel_size // 2
        ).astype(np.uint8)

        if min_region_size > 0:
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                if np.sum(labeled == i) < min_region_size:
                    mask[labeled == i] = 0

    return mask


def psr_coverage(mask: np.ndarray) -> float:
    """Calculate the percentage of PSR pixels in a mask."""
    return float(np.mean(mask) * 100)


def classify_image_by_mask(
    image: np.ndarray,
    dark_threshold: float = 15.0 / 255.0,
) -> str:
    """Quick classification of an image based on dark pixel percentage."""
    pct_dark = np.mean(image < dark_threshold) * 100

    if pct_dark > 70:
        return 'PSR'
    elif pct_dark > 30:
        return 'MIXED'
    else:
        return 'SUNLIT'
