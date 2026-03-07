"""Model architecture definition."""
from torch import nn
import torch
import torch.nn.functional as F
from helpers.hyperparams import hyperparams

class REINFORCE(nn.Module):
    """Model which does RL work."""

    def __init__(self)->None:
        """Initialize and define the layers."""
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        # 3 outputs for x, y, and radius per circle
        self.policy = nn.Linear(256, 3*hyperparams.NUM_CIRCLES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: The input image tensor of shape (batch_size, 1, img_size, img_size).
        Returns:
            A tensor of shape (batch_size, 3 * num_circles) containing the predicted parameters for each circle.
        """
        x = self.encoder(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        predictions = torch.tanh(self.policy(x)) * 0.5 + 0.5

        return predictions
