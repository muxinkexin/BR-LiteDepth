from __future__ import absolute_import, division, print_function

import os
import sys
import tempfile


if __name__ == "__main__":
    # Remove system library overrides before Python loads PyTorch or CUDA.
    if "LD_LIBRARY_PATH" in os.environ or "LD_PRELOAD" in os.environ:
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("LD_PRELOAD", None)
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "litemono-matplotlib"))

    from options import LiteMonoOptions
    from trainer import Trainer

    opts = LiteMonoOptions().parse()
    trainer = Trainer(opts)
    trainer.train()
