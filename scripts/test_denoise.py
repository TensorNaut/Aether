import cv2
import numpy as np

# Load the enhanced image
img = cv2.imread("output/enhanced/shackleton_test.png")

# Split into left (raw) and right (enhanced)
w = img.shape[1] // 2
raw = img[:, :w]
enhanced = img[:, w:]

# Apply Non-Local Means Denoising to the enhanced side
# The parameters (h) control the filter strength
denoised = cv2.fastNlMeansDenoising(enhanced, None, h=25, templateWindowSize=7, searchWindowSize=21)

# Apply CLAHE to the denoised image to pop the crater features
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
denoised_clahe = clahe.apply(cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY))
denoised_clahe = cv2.cvtColor(denoised_clahe, cv2.COLOR_GRAY2BGR)

# Concatenate: Raw | Zero-DCE | Denoised+CLAHE
final = np.concatenate([raw, enhanced, denoised_clahe], axis=1)

cv2.imwrite("output/enhanced/shackleton_denoised.png", final)
print("Saved shackleton_denoised.png")
