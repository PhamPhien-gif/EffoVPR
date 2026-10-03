import pytest
import torch

from src.effovpr.models.cosface import CosFace
from src.effovpr.models.dino_attention import assert_patch_count, attention_patch_scores, parse_tokens, resolve_layer_index
from src.effovpr.models.projection import GlobalProjection


def test_global_projection_shape():
    x = torch.randn(4, 1024)
    proj = GlobalProjection(1024, 128)
    y = proj(x)
    assert y.shape == (4, 128)


def test_cosface_loss_shape():
    emb = torch.randn(8, 128)
    labels = torch.randint(0, 4, (8,))
    cos = CosFace(128, 4)
    logits, loss = cos(emb, labels)
    assert logits.shape == (8, 4)
    assert loss.ndim == 0


def test_cosface_backward_has_finite_gradients():
    embeddings = torch.randn(3, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    classifier = CosFace(16, 3)
    _, loss = classifier(embeddings, labels)
    loss.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert classifier.weight.grad is not None and torch.isfinite(classifier.weight.grad).all()


def test_token_parsing_and_layer_resolution():
    tokens = torch.randn(2, 261, 1024)
    cls, register, patch = parse_tokens(tokens)
    assert cls.shape == (2, 1, 1024)
    assert register.shape[1] == 4
    assert patch.shape[1] == 256
    assert_patch_count(patch, image_resolution=224)
    high_resolution_patches = torch.randn(1, 1296, 1024)
    assert_patch_count(high_resolution_patches, image_resolution=504)
    assert resolve_layer_index("n-1", 24) == 23
    assert resolve_layer_index("n", 24) == 23


def test_attention_patch_scores_are_a_probability_distribution():
    query_patches = torch.randn(12, 8)
    cls_key = torch.randn(8)
    scores = attention_patch_scores(query_patches, cls_key)
    assert scores.shape == (12,)
    assert torch.all(scores >= 0)
    assert scores.sum().item() == pytest.approx(1.0)
