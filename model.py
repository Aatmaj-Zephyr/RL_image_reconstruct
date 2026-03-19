"""Model architecture definition."""
from torch import nn
import torch
import torch.nn.functional as F
from helpers.hyperparams import hyperparams


class REINFORCE(nn.Module):
    """Model which does RL work."""

    def __init__(self) -> None:
        """Initialize and define the layers."""
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.policy = nn.Linear(512, 3*hyperparams.NUM_CIRCLES+6*hyperparams.NUM_TRIANGLES)  # output layer for circle and triangle parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: The input image tensor of shape (batch_size, 1, img_size, img_size).
        Returns:
            A tensor of shape (batch_size, 3 * num_circles + 6 * num_triangles) containing the predicted parameters for each circle and triangle.
        """
        x = self.encoder(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))

        predictions = F.sigmoid(self.policy(x))

        return predictions
