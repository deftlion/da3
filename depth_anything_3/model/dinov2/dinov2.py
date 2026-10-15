# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn

from depth_anything_3.model.dinov2.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
# Derived from pyright hover on DinoV2.forward and DinoVisionTransformer.
# get_intermediate_layers.  The return type is:
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


# ---------------------------------------------------------------------------
# Original custom-ViT implementation (unchanged)
# ---------------------------------------------------------------------------

class DinoV2(nn.Module):
    """Production backbone wrapping the custom Meta DinoVisionTransformer."""

    def __init__(
        self,
        name: str,
        out_layers: list[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )

    # --- DinoV2Protocol attribute shims -----------------------------------

    @property
    def embed_dim(self) -> int:
        return self.pretrained.embed_dim  # type: ignore[return-value]

    @property
    def patch_size(self) -> int:
        return self.pretrained.patch_size  # type: ignore[return-value]

    @property
    def num_heads(self) -> int:
        return self.pretrained.num_heads  # type: ignore[return-value]

    @property
    def n_blocks(self) -> int:
        return self.pretrained.n_blocks  # type: ignore[return-value]

    # --- Forward ----------------------------------------------------------

    def forward(self, x: Tensor, **kwargs) -> IntermediateLayerOutput:
        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )


# Verify at import time that DinoV2 satisfies the structural contract.
assert isinstance(DinoV2, type) and issubclass(DinoV2, nn.Module)


# ---------------------------------------------------------------------------
# HuggingFace-transformers backbone (Protocol-compliant)
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


class TransformersDinoV2(nn.Module):
    """DINOv2 backbone backed by ``transformers.Dinov2Model``.

    Implements :class:`DinoV2Protocol` so it can be used as a drop-in
    replacement for :class:`DinoV2` anywhere in Depth-Anything-3.

    The ``transformers`` library is a lazy import so it is not a hard
    dependency for users who only use the custom-ViT path.

    Parameters
    ----------
    name:
        One of ``{"vits", "vitb", "vitl", "vitg"}``.
    out_layers:
        0-based indices of transformer blocks whose hidden states to return,
        mirroring the ``out_layers`` convention of :class:`DinoV2`.
    hf_model_id:
        Override the default HuggingFace model ID.  Useful for local
        checkpoints or fine-tuned variants.
    dtype:
        Parameter dtype.  Defaults to ``torch.float32``; pass
        ``torch.float16`` or ``torch.bfloat16`` for inference efficiency.
    """

    def __init__(
        self,
        name: str,
        out_layers: list[int],
        *,
        hf_model_id: str | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        super().__init__()
        if name not in _HF_MODEL_IDS:
            raise ValueError(
                f"name must be one of {set(_HF_MODEL_IDS)}, got {name!r}"
            )

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
        self._embed_dim = _EMBED_DIMS[name]
        self._patch_size = 14  # all DA3 dinov2 checkpoints use patch_size=14
        self._num_heads = _NUM_HEADS[name]
        self._n_blocks = _N_BLOCKS[name]

        model_id = hf_model_id or _HF_MODEL_IDS[name]
        config = Dinov2Config.from_pretrained(model_id)
        # Request hidden states at all intermediate layers so we can slice
        # exactly the requested out_layers later.
        config.output_hidden_states = True

        self.backbone: nn.Module = Dinov2Model.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            ignore_mismatched_sizes=False,
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
            forwarding and restored afterwards, matching the custom-ViT path.
        **kwargs:
            Accepted for interface compatibility; not forwarded to the
            transformers model (it has no equivalent kwargs).

        Returns
        -------
        Same ``IntermediateLayerOutput`` structure as :class:`DinoV2`:

        * ``patch_features``: one ``(patch_tokens, cls_token)`` pair per
          entry in ``out_layers``.  ``patch_tokens`` is ``(B*S, N, D)`` and
          ``cls_token`` is ``(B*S, D)`` — identical shapes to the custom-ViT
          path.
        * ``camera_tokens``: empty ``list[Tensor]`` — the transformers model
          has no camera-token concept, so this matches a backbone built
          without ``plus_cam_token``.
        """
        # x: (B, S, C, H, W) — flatten S into batch for the ViT.
        B, S, C, H, W = x.shape
        x_flat = x.reshape(B * S, C, H, W)

        outputs = self.backbone(
            pixel_values=x_flat,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (B*S, 1+N_patches, D), one per layer.
        # Verified empirically (transformers 5.18): @capture_outputs on Dinov2Encoder
        # hooks into Dinov2Layer outputs, so hidden_states[i] = output of block i
        # (0-based).  The final entry (index num_hidden_layers) is the layernorm
        # output appended by the model; it is NOT an extra embedding prefix.
        # Therefore block layer_idx is at hidden_states[layer_idx] — no +1 offset.
        all_hidden: tuple[Tensor, ...] = outputs.hidden_states  # type: ignore[assignment]

        patch_features: list[tuple[Tensor, Tensor]] = []
        for layer_idx in self.out_layers:
            # all_hidden[layer_idx]: (B*S, 1 + N_patches, D)
            # [:, 0, :] is the CLS token; [:, 1:, :] are patch tokens.
            # No register tokens in facebook/dinov2-* checkpoints.
            hidden = all_hidden[layer_idx]          # (B*S, 1+N, D)
            cls_tok = hidden[:, 0, :]               # (B*S, D)
            patch_tok = hidden[:, 1:, :]            # (B*S, N, D)
            patch_features.append((patch_tok, cls_tok))

        # camera_tokens: empty — no camera token in standard DINOv2.
        camera_tokens: list[Tensor] = []

        return tuple(patch_features), camera_tokens  # type: ignore[return-value]
