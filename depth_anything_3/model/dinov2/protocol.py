# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Structural type contract for DINOv2 backbones used inside Depth-Anything-3.

Import from here when you need the type only — no PyTorch ``nn`` or
``transformers`` baggage is pulled in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


# ---------------------------------------------------------------------------
# Return-type alias
# ---------------------------------------------------------------------------
# Derived from pyright hover on DinoV2.forward and
# DinoVisionTransformer.get_intermediate_layers.
#
#   tuple[
#       tuple[tuple[Tensor, Tensor], ...],   # per-requested-layer patch tokens
#       list[Tensor],                         # camera/aux tokens per layer
#   ]
#
# Consumers (e.g. DPT heads) call `model.dinov2.dinov2(x, ...)` and unpack
# exactly this structure.
# ---------------------------------------------------------------------------

IntermediateLayerOutput = tuple[tuple[tuple[Tensor, Tensor], ...], list[Tensor]]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DinoV2Protocol(Protocol):
    """Structural type for any DINOv2 backbone used inside Depth-Anything-3.

    The LSP-verified forward signature is::

        def forward(
            self,
            x: Tensor,               # (B, S, C, H, W)  fp16/bf16 image batch
            **kwargs,
        ) -> tuple[
            tuple[tuple[Tensor, Tensor], ...],  # patch features per layer
            list[Tensor],                        # camera/aux tokens per layer
        ]

    Concrete subclasses must also expose the scalar attributes read by the
    DPT / DualDPT heads when building feature-pyramid adapters.
    """

    # Scalar attributes inspected by downstream heads ----------------------

    @property
    def embed_dim(self) -> int:
        """Hidden dimension of the ViT (e.g. 384/768/1024/1536)."""
        ...

    @property
    def patch_size(self) -> int:
        """Spatial patch stride in pixels (always 14 for DA3 checkpoints)."""
        ...

    @property
    def num_heads(self) -> int:
        """Number of attention heads in each ViT block."""
        ...

    @property
    def n_blocks(self) -> int:
        """Total number of transformer blocks in the backbone."""
        ...

    # Call contract --------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        **kwargs,
    ) -> IntermediateLayerOutput:
        """Return intermediate patch features + camera/aux tokens.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, S, C, H, W)`` in fp16 or bf16,
            already normalised.
        **kwargs:
            Forwarded verbatim to ``DinoVisionTransformer.get_intermediate_layers``.
            Known keys: ``masks``, ``cls_token``, ``export_feat_layers``.

        Returns
        -------
        patch_features:
            One ``(patch_tokens, cls_token)`` pair per requested layer.
        camera_tokens:
            One aux-token tensor per requested layer (may be empty list).
        """
        ...
