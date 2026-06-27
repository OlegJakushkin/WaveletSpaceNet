"""WaveletSpaceNet — sparse context + a wavelet image pyramid -> mesh-plane + camera pose.

A scene-level sibling of *Points-as-(Super)Tori*: the same ``[context | SEP | main]``
Perceiver fusion and wavelet-emitting decoder, applied to a grayscale frame (encoded as
a multi-scale wavelet block pyramid) plus a sparse 3-D context cloud.
"""
from .model import WaveletSpaceNet, save_checkpoint, load_checkpoint, DIODE_LOG_STATS
from .wavelet2d import (WaveletPyramidTokenizer, haar_filters_2d, dwt2d, idwt2d, DEFAULT_LEVELS)
from . import geometry, diode, losses, blocks

__all__ = [
    "WaveletSpaceNet", "save_checkpoint", "load_checkpoint", "DIODE_LOG_STATS",
    "WaveletPyramidTokenizer", "haar_filters_2d", "dwt2d", "idwt2d", "DEFAULT_LEVELS",
    "geometry", "diode", "losses", "blocks",
]
