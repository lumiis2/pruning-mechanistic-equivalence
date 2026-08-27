import pytest
import torch

from src.models import SmallCNN


def test_model_logits_shape():
    model = SmallCNN(hidden_dim=32)
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 2)


def test_model_returns_requested_activations():
    model = SmallCNN(hidden_dim=32)
    logits, activations = model(torch.randn(2, 3, 32, 32), return_activations=True)
    assert logits.shape == (2, 2)
    assert set(activations) == {"conv1", "conv2", "hidden"}
    assert activations["conv1"].shape == (2, 8, 32, 32)
    assert activations["conv2"].shape == (2, 16, 16, 16)
    assert activations["hidden"].shape == (2, 32)


def test_model_supports_layer_activation_replacement():
    model = SmallCNN(hidden_dim=32)
    inputs = torch.randn(2, 3, 32, 32)
    _, activations = model(inputs, return_activations=True)
    replacement = torch.zeros_like(activations["hidden"])
    patched = model(inputs, activation_replacements={"hidden": replacement})
    expected = model.fc2(replacement)
    assert torch.allclose(patched, expected)


def test_model_rejects_unknown_activation_replacement():
    model = SmallCNN(hidden_dim=32)
    with pytest.raises(ValueError, match="unknown activation"):
        model(torch.randn(1, 3, 32, 32), activation_replacements={"other": torch.zeros(1)})
