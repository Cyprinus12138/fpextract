"""Optional learned watermark removal via SLBR (Self-calibrated Localization
and Background Refinement, ACM MM 2021).

Setup (not installed by pip)::

    git clone https://github.com/bcmi/SLBR-Visible-Watermark-Removal <somewhere>
    export FPEXTRACT_SLBR_ROOT=<somewhere>
    # download the pretrained weights from the SLBR repo and pass their path
    pip install 'fpextract[slbr]'

SLBR runs at 512×512, so its own output is blurry. This wrapper uses it only
to produce the watermark *mask*, then inpaints at native resolution with
OpenCV so unmasked pixels stay untouched.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from fpextract.cv import require_cv2


def slbr_root() -> Path:
    root = os.environ.get("FPEXTRACT_SLBR_ROOT")
    if not root:
        raise FileNotFoundError("set FPEXTRACT_SLBR_ROOT to a clone of SLBR-Visible-Watermark-Removal")
    return Path(root)


def _load_slbr_network(root: Path):
    """Load SLBR's network class without importing its training framework."""
    for pkg in ("src", "src.networks", "src.utils", "src.models"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(root / pkg.replace(".", "/"))]
            mod.__package__ = pkg
            sys.modules[pkg] = mod

    def _load_file(mod_name: str, rel: str):
        if mod_name in sys.modules and getattr(sys.modules[mod_name], "__spec__", None):
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, root / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_file("src.utils.model_init", "src/utils/model_init.py")
    _load_file("src.networks.blocks", "src/networks/blocks.py")
    return _load_file("src.networks.resunet", "src/networks/resunet.py").SLBR


class Dewatermarker:
    """Inference-only SLBR wrapper. Weights load lazily on first call."""

    _MODEL_SIZE = 512
    _INPAINT_R = 3

    def __init__(self, weights: str | Path, device: str = "cuda") -> None:
        self._weights = Path(weights)
        self._device = device
        self._net = None

    def _load(self) -> None:
        if self._net is not None:
            return
        if not self._weights.exists():
            raise FileNotFoundError(f"SLBR weights not found: {self._weights}")
        root = slbr_root()
        if not root.exists():
            raise FileNotFoundError(f"SLBR repo not found at {root}")
        import torch
        SLBR = _load_slbr_network(root)
        args = SimpleNamespace(use_refine=True, k_refine=3, k_skip_stage=3, bg_mode="res_mask",
                               mask_mode="cat", project_mode="simple", sim_metric="cos",
                               k_center=2, lr=1e-4)
        net = SLBR(args, shared_depth=1, blocks=3, long_skip=True)
        ckpt = torch.load(self._weights, map_location=self._device)
        net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=True)
        self._net = net.to(self._device).eval()

    def _forward(self, img_bgr):
        require_cv2()
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F

        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        x_in = F.interpolate(x, size=(self._MODEL_SIZE, self._MODEL_SIZE), mode="bilinear",
                             align_corners=False)
        with torch.no_grad():
            outputs = self._net(x_in)
            decoded_t, mask_t = outputs[0][0], outputs[1][0]
        dec = decoded_t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        dec = cv2.resize(dec, (w, h), interpolation=cv2.INTER_LINEAR)
        decoded_bgr = cv2.cvtColor(np.clip(dec * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask_full = cv2.resize(mask_t.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        mask_u8 = np.clip(mask_full * 255, 0, 255).astype(np.uint8)
        _, mask_bin = cv2.threshold(mask_u8, 200, 255, cv2.THRESH_BINARY)
        min_area = max(300, (h * w) // 800)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        filtered = np.zeros_like(mask_bin)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
        return decoded_bgr, cv2.dilate(filtered, np.ones((5, 5), np.uint8), iterations=2)

    def get_mask(self, img_bgr):
        self._load()
        return self._forward(img_bgr)[1]

    def __call__(self, img_bgr):
        import cv2
        import numpy as np
        self._load()
        _, mask = self._forward(img_bgr)
        if not np.any(mask):
            return img_bgr
        return cv2.inpaint(img_bgr, mask, self._INPAINT_R, cv2.INPAINT_TELEA)
