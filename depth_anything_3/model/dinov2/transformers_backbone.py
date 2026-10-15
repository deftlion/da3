# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""HuggingFace-transformers-backed DINOv2 backbone.

Implements :class:`~depth_anything_3.model.dinov2.protocol.DinoV2Protocol`
as a drop-in replacement for the custom-ViT
:class:`~depth_anything_3.model.dinov2.dinov2.DinoV2`.

The ``transformers`` library is a **lazy import** — it is not loaded until
``TransformersDinoV2.__init__`` is called, so importing this module does not
add ``transformers`` as a hard dependency for users of the custom-ViT path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from depth_anything_3.model.dinov2.protocol import (
    DinoV2Protocol,
    IntermediateLayerOutput,
)

# ---------------------------------------------------------------------------
# Per-variant constants (all verified against facebook/dinov2-* configs)
# ---------------------------------------------------------------------------

_HF_MODEL_IDS: dict[str, str] = {
    "vits": "facebook/dinov2-small",
    "vitb": "facebook/dinov2-base",
    "vitl": "facebook/dinov2-large",
    "vitg": "facebook/dinov2-giant",
}

_EMBED_DIMS: dict[str, int] = {
    "vits": 384,
    "vitb": 768,
    "vitl": 1024,
    "vitg": 1536,
}

_NUM_HEADS: dict[str, int] = {
    "vits": 6,
    "vitb": 12,
    "vitl": 16,
    "vitg": 24,
}

_N_BLOCKS: dict[str, int] = {
    "vits": 12,
    "vitb": 12,
    "vitl": 24,
    "vitg": 40,
}


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class TransformersDinoV2(nn.Module, DinoV2Protocol):
    """DINOv2 backbone backed by ``transformers.Dinov2Model``.

    Implements :class:`DinoV2Protocol` so it can be used as a drop-in
    replacement for :class:`~depth_anything_3.model.dinov2.dinov2.DinoV2`
    anywhere in Depth-Anything-3.

    Note
    ----
    Unlike the custom DA3 ViT in :mod:`depth_anything_3.model.dinov2.vision_transformer`,
    the HuggingFace ``Dinov2Model`` backbone uses standard learned 1D positional
    embeddings and standard ViT attention without 2D RoPE or QK-Norm.
    Parameters such as ``rope_start`` and ``qknorm_start`` are accepted for
    interface compatibility.

    Parameters
    ----------
    name:
        One of ``{"vits", "vitb", "vitl", "vitg"}``.
    out_layers:
        0-based indices of transformer blocks whose hidden states to return,
        mirroring the ``out_layers`` convention of ``DinoV2``.
    hf_model_id:
        Override the default HuggingFace model ID.  Useful for local
        checkpoints or fine-tuned variants.
    dtype:
        Parameter dtype.  Defaults to ``torch.float32``; pass
        ``torch.float16`` or ``torch.bfloat16`` for inference efficiency.
    attn_implementation:
        Attention implementation passed to ``Dinov2Model.from_pretrained``
        (e.g. ``"sdpa"``, ``"flash_attention_2"``, or ``None`` for default).
    """

    def __init__(
        self,
        name: str,
        out_layers: list[int],
        *,
        hf_model_id: str | None = None,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if name not in _HF_MODEL_IDS:
            raise ValueError(f"name must be one of {set(_HF_MODEL_IDS)}, got {name!r}")

        # Lazy import — keeps `transformers` optional for custom-ViT users.
        try:
            from transformers import Dinov2Config, Dinov2Model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TransformersDinoV2 requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from exc

        self.name = name
        self.out_layers = out_layers
        self.alt_start = kwargs.get("alt_start", -1)
        self.qknorm_start = kwargs.get("qknorm_start", -1)
        self.rope_start = kwargs.get("rope_start", -1)
        self.cat_token = kwargs.get("cat_token", True)
        self.extra_kwargs = kwargs
        self._embed_dim = _EMBED_DIMS[name]
        self._patch_size = 14  # all DA3 dinov2 checkpoints use patch_size=14
        self._num_heads = _NUM_HEADS[name]
        self._n_blocks = _N_BLOCKS[name]

        model_id = hf_model_id or _HF_MODEL_IDS[name]
        config = Dinov2Config.from_pretrained(model_id)
        # Set on config so it persists regardless of forward-call kwargs
        # (@capture_outputs reads config.output_hidden_states as fallback).
        config.output_hidden_states = True

        from_pretrained_kwargs: dict[str, object] = {
            "config": config,
            "torch_dtype": dtype,
            "ignore_mismatched_sizes": False,
        }
        if attn_implementation is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_implementation

        self.backbone: nn.Module = Dinov2Model.from_pretrained(
            model_id,
            **from_pretrained_kwargs,
        )

    # --- DinoV2Protocol attributes ----------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    # --- Forward ----------------------------------------------------------

    def forward(self, x: Tensor, **kwargs) -> IntermediateLayerOutput:
        """Extract intermediate DINOv2 features via the transformers backbone.

        Parameters
        ----------
        x:
            Float image tensor of shape ``(B, S, C, H, W)``.  The leading
            scene-stack dimension ``S`` is flattened into the batch before
            forwarding, matching the custom-ViT path.
        **kwargs:
            Accepted for interface compatibility; not forwarded to the
            transformers model.

        Returns
        -------
        patch_features:
            One ``(patch_tokens, cls_token)`` pair per entry in
            ``out_layers``.  ``patch_tokens`` shape ``(B*S, N, D)``;
            ``cls_token`` shape ``(B*S, D)`` — identical to the custom-ViT
            path.
        camera_tokens:
            Always ``[]`` — the transformers model has no camera-token
            concept, matching a backbone built without ``plus_cam_token``.
        """
        # Flatten scene-stack dimension into batch for the ViT.
        B, S, C, H, W = x.shape
        x_flat = x.reshape(B * S, C, H, W)

        outputs = self.backbone(
            pixel_values=x_flat,
            output_hidden_states=True,  # belt-and-suspenders alongside config
            return_dict=True,
        )

        # hidden_states: tuple of (B*S, 1+N_patches, D), one entry per layer.
        #
        # Verified empirically with transformers 5.18.0.dev0:
        # @capture_outputs on Dinov2Encoder hooks into Dinov2Layer outputs, so
        #   hidden_states[i]  = output of block i  (0-based, no +1 offset)
        #   hidden_states[-1] = layernorm output   (= last_hidden_state)
        #
        # Sequence layout: [CLS, patch_0, …, patch_{N-1}]
        # No register tokens in facebook/dinov2-* checkpoints.
        all_hidden: tuple[Tensor, ...] = outputs.hidden_states  # type: ignore[assignment]

        patch_features: list[tuple[Tensor, Tensor]] = []
        for layer_idx in self.out_layers:
            hidden = all_hidden[layer_idx]  # (B*S, 1+N, D)
            cls_tok = hidden[:, 0, :]  # (B*S, D)
            patch_tok = hidden[:, 1:, :]  # (B*S, N, D)
            patch_features.append((patch_tok, cls_tok))

        camera_tokens: list[Tensor] = []
        return tuple(patch_features), camera_tokens  # type: ignore[return-value]


# Verify the structural contract at import time (no instantiation required).
assert issubclass(TransformersDinoV2, nn.Module)
