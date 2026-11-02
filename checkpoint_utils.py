"""Safe loading for the legacy Lite-Mono ImageNet pretraining checkpoint."""

import argparse

import numpy as np
import torch


def load_pretraining_checkpoint(path):
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core

    safe_globals = [
        (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
        argparse.Namespace,
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)
