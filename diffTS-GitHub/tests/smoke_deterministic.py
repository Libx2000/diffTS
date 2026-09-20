from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from diffts.model import UNet3D  # noqa: E402


def main() -> None:
    model = UNet3D(in_channels=8, out_channels=2, base=8, levels=3, dropout=0.0)
    x = torch.randn(1, 8, 15, 120, 240)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 15, 120, 240), y.shape
    print("3D U-Net 形状测试通过：", tuple(y.shape))


if __name__ == "__main__":
    main()
