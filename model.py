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
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy = nn.Linear(64, 3*hyperparams.NUM_CIRCLES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: The input image tensor of shape (batch_size, 1, img_size, img_size).
        Returns:
            A tensor of shape (batch_size, 3 * num_circles) containing the predicted parameters for each circle.
        """
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        predictions = F.sigmoid(self.policy(x))

        return predictions
