import os, torch, json
from collections import OrderedDict
from typing import Dict, Any


class CheckpointManager:
    """
    save_path/ckpt_xxxx.pth  (二進位)
    save_path/ckpt_xxxx.json (迭代號 ↔ 路徑對照，可省)
    """
    def __init__(self):
        pass

    # ---------- public API ----------
    def save(self,
             iteration:      int,
             models:         Dict[str, torch.nn.Module],
             optimizers:     Dict[str, torch.optim.Optimizer],
             path:           str = None):
        """
        models  = {"gen": G, "disc": D, "tD": templateD}
        optimizers 同名對應
        extra 可附加其他超參、label-map
        """
        state = OrderedDict(iter=iteration)
        state.update({f"{k}_state": v.state_dict() for k, v in models.items()})
        state.update({f"opt_{k}_state": v.state_dict() for k, v in optimizers.items()})

        torch.save(state, path)
        print(f"[Checkpoint] saved → {path}")

    def load(self,
             path: str,
             models: Dict[str, torch.nn.Module],
             optimizers: Dict[str, torch.optim.Optimizer] = None,
             map_location="cpu"):
        state = torch.load(path, map_location=map_location)

        for k, m in models.items():
            m.load_state_dict(state[f"{k}_state"], strict=False)

        if optimizers is not None:
            for k, opt in optimizers.items():
                key = f"opt_{k}_state"
                if key in state:
                    opt.load_state_dict(state[key])

        start_iter = state.get("iter", 0)
        print(f"[Checkpoint] resume from {path} @ iter {start_iter}")
        return start_iter
