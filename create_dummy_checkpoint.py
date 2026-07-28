from __future__ import annotations

from pathlib import Path

import torch

from inspired_risk.modeling import TinyDRNet


def main() -> None:
    out = Path("checkpoints/dummy_tinydrnet.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = TinyDRNet()
    torch.save(model.state_dict(), out)
    print(f"Saved demo checkpoint to {out}")


if __name__ == "__main__":
    main()
