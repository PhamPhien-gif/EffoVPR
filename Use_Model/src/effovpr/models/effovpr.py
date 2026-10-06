from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cosface import CosFace
from .dino_attention import assert_patch_count, attention_patch_scores, parse_tokens, resolve_layer_index
from .dino_backbone import DINOv2Backbone
from .projection import GlobalProjection


class EffoVPR(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-with-registers-large",
        global_dim: int = 1024,
        num_classes: Optional[int] = None,
        fine_tune_last_n_layers: int = 5,
        with_registers: bool = True,
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.4,
        cache_dir: str | None = None,
        backbone_config: dict | None = None,
        processor_config: dict | None = None,
    ):
        super().__init__()
        self.backbone = DINOv2Backbone(model_name=backbone_name, with_registers=with_registers, cache_dir=cache_dir, backbone_config=backbone_config, processor_config=processor_config)
        self.global_dim = global_dim
        self.global_projection = GlobalProjection(input_dim=self.backbone.hidden_dim, output_dim=global_dim)
        self.num_classes = num_classes
        self.cosface = CosFace(in_features=global_dim, out_features=num_classes, scale=cosface_scale, margin=cosface_margin) if num_classes is not None else None
        self.configure_trainable_layers(last_n_layers=fine_tune_last_n_layers)

    def configure_trainable_layers(self, last_n_layers: int = 5):
        self.fine_tune_last_n_layers = last_n_layers
        if last_n_layers < 0 or last_n_layers > self.backbone.num_layers:
            raise ValueError(f"last_n_layers must be between 0 and {self.backbone.num_layers}")
        self.backbone.configure_trainable_layers(last_n_layers=last_n_layers)
        for param in self.global_projection.parameters():
            param.requires_grad = True
        if self.cosface is not None:
            for param in self.cosface.parameters():
                param.requires_grad = True

    def _prepare_images(self, images):
        if isinstance(images, torch.Tensor):
            return images
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
            return torch.stack(images, dim=0)
        processed = self.backbone.image_processor(images=images, return_tensors="pt")
        return processed["pixel_values"]

    def _apply_projection(self, cls_token: torch.Tensor) -> torch.Tensor:
        feat = self.global_projection(cls_token)
        return F.normalize(feat, p=2, dim=-1)

    def extract_global(self, image):
        pixel_values = self._prepare_images(image)
        pixel_values = pixel_values.to(next(self.parameters()).device)
        outputs = self.backbone(pixel_values)
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, 0, :]
        return self._apply_projection(cls_token)

    def extract_global_and_local(self, image, layer: int | str = "n-1"):
        pixel_values = self._prepare_images(image)
        pixel_values = pixel_values.to(next(self.parameters()).device)
        outputs = self.backbone(pixel_values)
        hidden_states = outputs.hidden_states
        layer_index = resolve_layer_index(layer, self.backbone.num_layers)
        selected = hidden_states[layer_index + 1]
        cls_token, register_tokens, patch_tokens = parse_tokens(selected)
        patch_tokens = patch_tokens.reshape(pixel_values.size(0), -1, patch_tokens.size(-1))
        assert_patch_count(patch_tokens, image_resolution=int(pixel_values.shape[-1]) if pixel_values.shape[-1] % self.backbone.patch_size == 0 else None)
        global_feat = self._apply_projection(outputs.last_hidden_state[:, 0, :])
        return global_feat, {"layer": layer_index, "cls_token": cls_token, "register_tokens": register_tokens, "patch_tokens": patch_tokens}

    def extract_qkv(self, image, layer: int | str = "n-1"):
        pixel_values = self._prepare_images(image)
        pixel_values = pixel_values.to(next(self.parameters()).device)
        outputs = self.backbone(pixel_values)
        hidden_states = outputs.hidden_states
        layer_index = resolve_layer_index(layer, self.backbone.num_layers)
        input_tokens = hidden_states[layer_index]
        block = self.backbone.encoder_layers()[layer_index]
        if hasattr(block, "norm1"):
            attention_input = block.norm1(input_tokens)
        elif hasattr(block, "layernorm_before"):
            attention_input = block.layernorm_before(input_tokens)
        else:
            raise RuntimeError(f"Cannot locate pre-attention normalization in {type(block).__name__}")
        attention = getattr(block, "attention", getattr(block, "self_attn", None))
        if attention is None:
            raise RuntimeError(f"Cannot locate attention module in {type(block).__name__}")
        # Hugging Face's DINOv2 implementations expose either q_proj/k_proj/v_proj
        # directly or query/key/value under an inner attention module.
        projections = [getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj")]
        if all(projection is not None for projection in projections):
            q_proj, k_proj, v_proj = projections
        else:
            inner_attention = getattr(attention, "attention", attention)
            projections = [getattr(inner_attention, name, None) for name in ("query", "key", "value")]
            if not all(projection is not None for projection in projections):
                raise RuntimeError(f"Unsupported Q/K/V projection structure in {type(attention).__name__}")
            q_proj, k_proj, v_proj = projections
        q = q_proj(attention_input)
        k = k_proj(attention_input)
        v = v_proj(attention_input)
        cls_token, register_tokens, _ = parse_tokens(q)
        v_patch_tokens = parse_tokens(v)[2]
        assert_patch_count(v_patch_tokens, image_resolution=int(pixel_values.shape[-1]) if pixel_values.shape[-1] % self.backbone.patch_size == 0 else None)
        return {
            "q": q,
            "k": k,
            "v": v,
            "cls_token": cls_token,
            "register_tokens": register_tokens,
            "q_patch": parse_tokens(q)[2],
            "k_patch": parse_tokens(k)[2],
            "v_patch": parse_tokens(v)[2],
            "layer": layer_index,
        }

    def compute_attention_score(self, qkv: Dict[str, torch.Tensor], T1: float = 0.05):
        q_patch = qkv["q_patch"][0]
        k_cls = qkv["k"][0, 0, :]
        scores = attention_patch_scores(q_patch, k_cls)
        selected = scores > T1
        selected_v = qkv["v_patch"][0][selected]
        return scores, selected, selected_v

    def forward(self, images, labels=None, **kwargs):
        pixel_values = self._prepare_images(images)
        outputs = self.backbone(pixel_values)
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, 0, :]
        global_feat = self._apply_projection(cls_token)
        if labels is not None and self.cosface is not None:
            logits, loss = self.cosface(global_feat, labels)
            return global_feat, logits, loss
        return global_feat
