import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.zero_dce import ZeroDCE
from src.data_io.pds4_reader import load_ohrc_image


class PSREnhancePipeline:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Zero-DCE Inference Pipeline on {self.device}...")
        
        # Load model
        self.model = ZeroDCE().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
    def enhance_crop(self, image_path, y_start, y_end, x_start, x_end, tile_size=512, overlap=64):
        """Enhances a specific crop of the massive OHRC image using overlapping tiles."""
        print(f"\nLoading raw image: {Path(image_path).name}")
        full_img = load_ohrc_image(image_path)
        
        # Extract the requested crop
        crop = full_img[y_start:y_end, x_start:x_end]
        h, w = crop.shape
        print(f"Extracted crop of size {w}x{h}")
        
        # Pad image to be divisible by tile_size
        pad_h = (tile_size - (h % tile_size)) % tile_size
        pad_w = (tile_size - (w % tile_size)) % tile_size
        
        if pad_h > 0 or pad_w > 0:
            crop = np.pad(crop, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        padded_h, padded_w = crop.shape
        enhanced_crop = np.zeros_like(crop)
        weight_map = np.zeros_like(crop)
        
        # Generate 2D Bartlett (tent) window for blending
        window_1d = np.bartlett(tile_size)
        window_2d = np.outer(window_1d, window_1d)
        
        step = tile_size - overlap
        
        y_steps = range(0, padded_h - tile_size + 1, step)
        x_steps = range(0, padded_w - tile_size + 1, step)
        
        total_tiles = len(list(y_steps)) * len(list(x_steps))
        print(f"Processing {total_tiles} tiles...")
        
        with torch.no_grad():
            pbar = tqdm(total=total_tiles)
            for y in y_steps:
                for x in x_steps:
                    # Extract tile
                    tile = crop[y:y+tile_size, x:x+tile_size]
                    
                    # Convert to tensor
                    tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # Enhance
                    enhanced_tensor, _ = self.model(tile_tensor)
                    enhanced_tile = enhanced_tensor.squeeze().cpu().numpy()
                    
                    # Blend into output using Bartlett window
                    enhanced_crop[y:y+tile_size, x:x+tile_size] += enhanced_tile * window_2d
                    weight_map[y:y+tile_size, x:x+tile_size] += window_2d
                    
                    pbar.update(1)
            pbar.close()
            
        # Normalize by weight map
        weight_map = np.clip(weight_map, 1e-8, None)
        enhanced_crop /= weight_map
        
        # Remove padding
        final_enhanced = enhanced_crop[:h, :w]
        original_crop = full_img[y_start:y_end, x_start:x_end]
        
        # Free memory
        del full_img
        
        return original_crop, final_enhanced

    def save_comparison(self, original, enhanced, output_path, stretch_raw=5.0):
        """Saves a side-by-side comparison image and a false-color version."""
        # Raw is usually so dark it's invisible, apply a simple linear stretch just for the visualization
        orig_vis = np.clip(original * stretch_raw, 0, 1)
        orig_8bit = (orig_vis * 255).astype(np.uint8)
        
        enh_8bit = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        
        # 1. Post-process: Non-Local Means Denoising + CLAHE
        # This fixes the "grainy grey" look by smoothing noise while preserving edges
        print("Applying Non-Local Means Denoising...")
        denoised = cv2.fastNlMeansDenoising(enh_8bit, None, h=25, templateWindowSize=7, searchWindowSize=21)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        final_clean = clahe.apply(denoised)
        
        # Concatenate: Raw | Zero-DCE | Denoised
        comparison = np.concatenate([orig_8bit, enh_8bit, final_clean], axis=1)
        
        # Save Grayscale Comparison
        img = Image.fromarray(comparison)
        img.save(output_path)
        print(f"Saved grayscale comparison to {output_path}")
        
        # 2. Layman/Publication Visual: False Color Map (Inferno)
        # This makes it very easy for a layman to see "bright = high signal, dark = low signal"
        color_map = cv2.applyColorMap(final_clean, cv2.COLORMAP_INFERNO)
        
        # Create a side-by-side of Raw vs False Color
        orig_color = cv2.cvtColor(orig_8bit, cv2.COLOR_GRAY2BGR)
        color_comparison = np.concatenate([orig_color, color_map], axis=1)
        
        color_path = str(output_path).replace('.png', '_color.png')
        cv2.imwrite(color_path, color_comparison)
        print(f"Saved false-color version to {color_path}")
