# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Compatibility alias for TransformersDinoV2.

Re-exports :class:`~depth_anything_3.model.dinov2.transformers_backbone.TransformersDinoV2`
from the canonical :mod:`depth_anything_3.model.dinov2.transformers_backbone` module.
"""

from depth_anything_3.model.dinov2.transformers_backbone import (
    _EMBED_DIMS,
    _HF_MODEL_IDS,
    _N_BLOCKS,
    _NUM_HEADS,
    TransformersDinoV2,
)

__all__ = [
    "TransformersDinoV2",
    "_EMBED_DIMS",
    "_HF_MODEL_IDS",
    "_N_BLOCKS",
    "_NUM_HEADS",
]
