from __future__ import annotations


import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv2Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-large",
        with_registers: bool = True,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.with_registers = with_registers
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.hidden_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_layers = len(self.encoder_layers())

    def encoder_layers(self):
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layer", getattr(encoder, "layers", None))
            if layers is not None:
                return layers
        vit = getattr(self.model, "vit", None)
        encoder = getattr(vit, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layers", getattr(encoder, "layer", None))
            if layers is not None:
                return layers
        raise RuntimeError(f"Unsupported DINOv2 encoder structure for {type(self.model).__name__}")

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        return self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True, **kwargs)

    def configure_trainable_layers(self, last_n_layers: int = 5):
        if last_n_layers < 0 or last_n_layers > self.num_layers:
            raise ValueError(f"last_n_layers must be between 0 and {self.num_layers}")
        for param in self.model.parameters():
            param.requires_grad = False
        layers = self.encoder_layers()
        for layer in list(layers)[-last_n_layers:] if last_n_layers else []:
            for param in layer.parameters():
                param.requires_grad = True
