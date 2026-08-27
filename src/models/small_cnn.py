"""Small CNN used by the Shape--Color sanity check."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class SmallCNN(nn.Module):
    """Two-convolution classifier with optional intermediate activations."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
        activation_replacements: Mapping[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run the classifier, optionally replacing named post-ReLU activations.

        Replacement is intentionally layer-level. It supports controlled
        activation patching without hooks or mutation of model parameters.
        """
        replacements = activation_replacements or {}
        unknown = set(replacements) - {"conv1", "conv2", "hidden"}
        if unknown:
            raise ValueError(f"unknown activation replacement(s): {sorted(unknown)}")

        def replace(name: str, activation: Tensor) -> Tensor:
            if name not in replacements:
                return activation
            replacement = replacements[name]
            if replacement.shape != activation.shape:
                raise ValueError(
                    f"replacement for {name} has shape {tuple(replacement.shape)}; "
                    f"expected {tuple(activation.shape)}"
                )
            return replacement

        conv1 = torch.relu(self.conv1(x))
        conv1 = replace("conv1", conv1)
        x = self.pool(conv1)
        conv2 = torch.relu(self.conv2(x))
        conv2 = replace("conv2", conv2)
        x = self.pool(conv2)
        x = torch.flatten(x, start_dim=1)
        hidden = torch.relu(self.fc1(x))
        hidden = replace("hidden", hidden)
        logits = self.fc2(hidden)

        if return_activations:
            return logits, {"conv1": conv1, "conv2": conv2, "hidden": hidden}
        return logits
