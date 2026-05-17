import torch
import numpy as np
import cv2
import piq
import pyiqa

def compute_metrics(raw_img_path, enhanced_img_path):
    # Load images as grayscale float32 [0, 1]
    raw_img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    enh_img = cv2.imread(enhanced_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # Convert to PyTorch tensors (1, 1, H, W)
    raw_tensor = torch.from_numpy(raw_img).unsqueeze(0).unsqueeze(0)
    enh_tensor = torch.from_numpy(enh_img).unsqueeze(0).unsqueeze(0)

    # 1. NIQE (Natural Image Quality Evaluator) - Lower is better
    # We use pyiqa as it has a robust implementation
    niqe_metric = pyiqa.create_metric('niqe', as_loss=False)
    # PyIQA expects 3 channels usually, so we duplicate
    enh_rgb = enh_tensor.repeat(1, 3, 1, 1)
    try:
        niqe_score = niqe_metric(enh_rgb).item()
    except:
        niqe_score = float('nan')

    # 2. BRISQUE - Lower is better
    try:
        brisque_metric = pyiqa.create_metric('brisque', as_loss=False)
        brisque_score = brisque_metric(enh_rgb).item()
    except:
        brisque_score = float('nan')

    # 3. Delta SNR (dB)
    # SNR = mean / std. We measure how much the signal-to-noise ratio increased.
    def get_snr(img):
        mean = np.mean(img)
        std = np.std(img)
        if std == 0: return 0
        return 20 * np.log10(mean / std)
    
    snr_raw = get_snr(raw_img)
    snr_enh = get_snr(enh_img)
    delta_snr = snr_enh - snr_raw

    # 4. PIQE (Perception based Image Quality Evaluator) - Lower is better
    try:
        piqe_metric = pyiqa.create_metric('piqe', as_loss=False)
        piqe_score = piqe_metric(enh_rgb).item()
    except:
        piqe_score = float('nan')

    return {
        'NIQE': niqe_score,
        'BRISQUE': brisque_score,
        'PIQE': piqe_score,
        'Delta_SNR': delta_snr,
        'Raw_SNR': snr_raw,
        'Enh_SNR': snr_enh
    }

if __name__ == '__main__':
    # Test on Shackleton 1
    import os
    from pathlib import Path
    
    output_dir = Path("output/enhanced")
    
    # The pipeline outputs concatenated images: [raw | enhanced_raw | enhanced_denoised]
    # We need to split them to evaluate
    shackleton_final = cv2.imread(str(output_dir / "shackleton_01_final.png"), cv2.IMREAD_GRAYSCALE)
    
    if shackleton_final is not None:
        w = shackleton_final.shape[1] // 3
        raw = shackleton_final[:, :w]
        denoised = shackleton_final[:, 2*w:]
        
        cv2.imwrite("temp_raw.png", raw)
        cv2.imwrite("temp_enh.png", denoised)
        
        metrics = compute_metrics("temp_raw.png", "temp_enh.png")
        print("\nShackleton 01 Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
        
        os.remove("temp_raw.png")
        os.remove("temp_enh.png")
    else:
        print("Image not found.")
