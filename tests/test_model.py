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
