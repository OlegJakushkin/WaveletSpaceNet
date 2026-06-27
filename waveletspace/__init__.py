"""WaveletSpaceNet v2 — sparse context + a learned image embedding -> mesh-plane + pose.

A scene-level sibling of *Points-as-(Super)Tori*: the same ``[context | SEP | image]``
Perceiver fusion and wavelet-emitting decoder, but with a learned conv-FPN image
embedding (:mod:`waveletspace.encoder`) and a multi-level Haar mesh-plane decoder tuned
to resolve small surfaces.
"""
from .model import WaveletSpaceNet, save_checkpoint, load_checkpoint, DIODE_LOG_STATS
from .encoder import EdgeFPNTokenizer, EdgeChannels, windowed_neighbors
from .wavelet2d import (haar_filters_2d, dwt2d, idwt2d, haar_analysis, haar_synthesis,
                        DEFAULT_LEVELS)
from . import geometry, diode, losses, blocks, encoder

__all__ = [
    "WaveletSpaceNet", "save_checkpoint", "load_checkpoint", "DIODE_LOG_STATS",
    "EdgeFPNTokenizer", "EdgeChannels", "windowed_neighbors",
    "haar_filters_2d", "dwt2d", "idwt2d", "haar_analysis", "haar_synthesis", "DEFAULT_LEVELS",
    "geometry", "diode", "losses", "blocks", "encoder",
]
