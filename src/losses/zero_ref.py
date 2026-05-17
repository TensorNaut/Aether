"""
Zero-Reference Loss Functions
==============================
Self-supervised losses for Zero-DCE training.

These losses require NO reference (ground-truth) images — critical
for PSR images where no "bright" version exists.

Reference: plan.pdf, Chapter 4, Equations 4.8-4.10
           Guo et al., CVPR 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """Preserves local contrast relationships during enhancement.

    Ensures that the relative brightness between a pixel and its
    4-neighbours is maintained after enhancement. If pixel A was
    brighter than pixel B in the input, it should stay brighter
    in the output.

    L_spa = (1/|N|) Σ_n ||(Ŷ - Ŷ_n) - (I - I_n)||²

    Reference: Eq. 4.8 in plan.pdf
    """

    def __init__(self):
        super().__init__()
        # Kernels for computing differences with 4-neighbours
        # Left, Right, Up, Down
        self.register_buffer(
            'kernel_left',
            torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
            .unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            'kernel_right',
            torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
            .unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            'kernel_up',
            torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
            .unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            'kernel_down',
            torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(
        self, original: torch.Tensor, enhanced: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            original: Input image (B, C, H, W)
            enhanced: Enhanced image (B, C, H, W)

        Returns:
            Scalar loss value
        """
        # Compute mean across channels for multi-channel images
        org_mean = torch.mean(original, dim=1, keepdim=True)
        enh_mean = torch.mean(enhanced, dim=1, keepdim=True)

        # Pool to reduce computation (average over 4x4 patches)
        org_pool = F.avg_pool2d(org_mean, kernel_size=4, stride=4)
        enh_pool = F.avg_pool2d(enh_mean, kernel_size=4, stride=4)

        # Compute spatial differences for each direction
        loss = 0.0
        for kernel in [self.kernel_left, self.kernel_right,
                       self.kernel_up, self.kernel_down]:
            d_org = F.conv2d(org_pool, kernel, padding=1)
            d_enh = F.conv2d(enh_pool, kernel, padding=1)
            loss += torch.mean((d_org - d_enh) ** 2)

        return loss / 4.0


class ExposureControlLoss(nn.Module):
    """Drives mean patch brightness toward a target exposure level.

    Divides the enhanced image into non-overlapping patches and
    penalises deviation of each patch's mean from a target E.

    L_exp = (1/M) Σ_k |Ŷ_k - E|

    Reference: Eq. 4.9 in plan.pdf

    Args:
        patch_size: Size of non-overlapping patches (default 16)
        E: Target mean exposure level (default 0.6)
           0.6 works well for general images; for lunar PSR images,
           we may want 0.4-0.5 to avoid over-brightening.
    """

    def __init__(self, patch_size: int = 16, E: float = 0.6):
        super().__init__()
        self.patch_size = patch_size
        self.E = E

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced: Enhanced image (B, C, H, W)

        Returns:
            Scalar loss value
        """
        # Average pool to get mean brightness per patch
        mean_patches = F.avg_pool2d(
            enhanced,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Penalise deviation from target exposure
        loss = torch.mean(torch.abs(mean_patches - self.E))
        return loss


class IlluminationSmoothnessLoss(nn.Module):
    """Total variation loss on the curve maps A(t).

    The enhancement curves should be spatially smooth to avoid
    artifacts. We penalise large gradients in the curve maps.

    L_TV = (1/HW) Σ_{i,j} √((∇_h A)² + (∇_v A)² + ε)

    Reference: Eq. 4.10 in plan.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, curve_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curve_maps: Curve parameter maps (B, n_iter*C, H, W)

        Returns:
            Scalar loss value
        """
        B, C, H, W = curve_maps.shape

        # Horizontal gradient
        grad_h = torch.abs(curve_maps[:, :, :, :-1] - curve_maps[:, :, :, 1:])

        # Vertical gradient
        grad_v = torch.abs(curve_maps[:, :, :-1, :] - curve_maps[:, :, 1:, :])

        # Mean total variation
        loss = torch.mean(grad_h) + torch.mean(grad_v)
        return loss


class ZeroDCELoss(nn.Module):
    """Combined Zero-DCE loss for self-supervised training.

    Weighted combination of spatial consistency, exposure control,
    and illumination smoothness losses.

    L_total = w_spa * L_spa + w_exp * L_exp + w_tv * L_tv

    Default weights from plan.pdf Eq. 4.11:
        w_spa = 1.0, w_exp = 10.0, w_tv = 200.0
    """

    def __init__(
        self,
        w_spa: float = 1.0,
        w_exp: float = 10.0,
        w_tv: float = 200.0,
        E: float = 0.6,
        patch_size: int = 16,
    ):
        super().__init__()
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_tv = w_tv

        self.spatial_loss = SpatialConsistencyLoss()
        self.exposure_loss = ExposureControlLoss(patch_size=patch_size, E=E)
        self.smoothness_loss = IlluminationSmoothnessLoss()

    def forward(
        self,
        original: torch.Tensor,
        enhanced: torch.Tensor,
        curve_maps: torch.Tensor,
    ) -> tuple:
        """
        Args:
            original: Input image (B, C, H, W)
            enhanced: Enhanced output (B, C, H, W)
            curve_maps: Predicted curve maps (B, n_iter*C, H, W)

        Returns:
            total_loss: Weighted sum (scalar)
            loss_dict: Individual loss values for logging
        """
        l_spa = self.spatial_loss(original, enhanced)
        l_exp = self.exposure_loss(enhanced)
        l_tv = self.smoothness_loss(curve_maps)

        total = (
            self.w_spa * l_spa
            + self.w_exp * l_exp
            + self.w_tv * l_tv
        )

        loss_dict = {
            'total': total.item(),
            'spatial': l_spa.item(),
            'exposure': l_exp.item(),
            'smoothness': l_tv.item(),
        }

        return total, loss_dict


if __name__ == '__main__':
    """Sanity check: verify losses compute and have reasonable magnitude."""
    print("Zero-Reference Loss Sanity Check")
    print("=" * 40)

    # Simulate a dark PSR image and its enhanced version
    B, C, H, W = 2, 1, 64, 64
    original = torch.rand(B, C, H, W) * 0.05  # Very dark input
    enhanced = torch.rand(B, C, H, W) * 0.6    # Brighter output
    curve_maps = torch.randn(B, 8, H, W) * 0.1  # 8 curve maps

    # Test individual losses
    spa = SpatialConsistencyLoss()
    exp = ExposureControlLoss(E=0.6)
    tv = IlluminationSmoothnessLoss()

    print(f"Spatial consistency: {spa(original, enhanced):.4f}")
    print(f"Exposure control:   {exp(enhanced):.4f}")
    print(f"Illum. smoothness:  {tv(curve_maps):.4f}")

    # Test combined loss
    combined = ZeroDCELoss()
    total, losses = combined(original, enhanced, curve_maps)
    print(f"\nCombined loss: {total:.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    # Verify gradients flow
    enhanced.requires_grad_(True)
    total, _ = combined(original, enhanced, curve_maps)
    total.backward()
    print(f"\nGradient flow: ✅ OK")
    print("=" * 40)
