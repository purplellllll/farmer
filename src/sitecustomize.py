"""Optional CUDA bootstrap for Ray workers on Windows.

Ray workers can import RLlib before Torch.  With some Windows CUDA builds that
ordering makes ``c10.dll`` fail to initialize.  Long-running training launchers
set ``FARMER_RL_PRELOAD_TORCH=1`` so every worker imports Torch at interpreter
startup; normal project and user Python processes are unaffected.
"""

from __future__ import annotations

import os


if os.environ.get("FARMER_RL_PRELOAD_TORCH") == "1":
    import torch  # noqa: F401
