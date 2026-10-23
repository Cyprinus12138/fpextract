"""Background normalisation via the black-hat morphological transform.

Any floor plan (gradient, branded or tinted background, semi-transparent
watermark) becomes a white-background image with dark line work preserved::

    blackhat   = close(channel, kernel) - channel
    normalised = 255 - blackhat
      background pixels            → 255 (white)
      solid wall pixels            → dark (preserved)
      semi-transparent watermark   → near-white (suppressed)
"""
from __future__ import annotations

from fpextract.cv import require_cv2


def normalize_background(img_bgr, bg_kernel_frac: float = 0.07):
    """Return a white-background BGR image. ``bg_kernel_frac`` is the kernel
    size as a fraction of the shorter side; 7% spans a typical room."""
    require_cv2()
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    r = max(31, (int(min(h, w) * bg_kernel_frac) // 2) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    out = np.empty_like(img_bgr)
    for c in range(3):
        out[:, :, c] = 255 - cv2.morphologyEx(img_bgr[:, :, c], cv2.MORPH_BLACKHAT, kernel)
    return out
