"""Environment for the RL agent."""
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from helpers.hyperparams import hyperparams
from helpers.logger import log


class ShapeDrawEnv(gym.Env):
    """Environment for the RL agent."""

    def __init__(self) -> None:
        """Initialize the agent following the inherited properties.
        """
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(
            3 * hyperparams.NUM_CIRCLES,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(hyperparams.IMG_SIZE, hyperparams
                                  .IMG_SIZE), dtype=np.float32)
        self.MESHGRID = torch.meshgrid(
            torch.arange(hyperparams.IMG_SIZE, device=hyperparams.DEVICE),
            torch.arange(hyperparams.IMG_SIZE, device=hyperparams.DEVICE),
            indexing="ij",
        )  # meshgrid for geometry calculations

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        """Reset the environment and sample new parameters for the circles.
        Args:
            Dont pass any args 
        Returns:
            gt_mask (torch.Tensor): Ground truth mask for the new episode
            info (dict): Dictionary containing GT parameters for logging and analysis
        """
        super().reset(seed=seed)
        self.gt_params = self._sample_environment_params()
        self.gt_mask = self._create_shape_masks(
            self.gt_params.to(hyperparams.DEVICE))
        info = {
            "gt_params": self.gt_params.cpu()
        }
        return self.gt_mask.cpu(), info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        """Take an action and return the predicted mask, reward, done flag, and info."""
        pred_mask = self._create_shape_masks(action)
        # Compute reward (IoU) for display
        reward = self._compute_reward(pred_mask, self.gt_mask)
        return pred_mask.cpu(), reward, False, False, {}

    def _compute_reward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Compute IoU reward for a single predicted and ground truth masks.
        Args:
            pred_mask (torch.Tensor): (H, W) predicted binary masks
            gt_mask (torch.Tensor): (H, W) ground truth binary masks
        Returns:
            reward (float): reward value for the sampled prediction
        """
        assert pred_mask.shape == gt_mask.shape, "Predicted and GT masks must have the same shape"

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection + 1e-6

        reward = intersection / union
        log.debug(f"Reward computed over batch: {reward}")

        return reward.item()

    def _sample_environment_params(self) -> torch.Tensor:
        """Sample random circle parameters for the environment."""
        circles = [torch.rand(3) for _ in range(hyperparams.NUM_CIRCLES)]
        return torch.cat(circles)

    def _create_shape_masks(self, circle_parameters: torch.Tensor) -> torch.Tensor:
        """Create a combined mask for multiple circles given their parameters.
        Args:
            circle_parameters (torch.Tensor): (3*num_circles) tensor of circle parameters
        Returns:
            torch.Tensor: (H, W) combined binary mask for all circles
        """
        circle_parameters = torch.tensor(circle_parameters, device="cpu")

        size = hyperparams.IMG_SIZE
        mask = torch.zeros((size, size),
                           device=hyperparams.DEVICE)

        for circle_idx in range(hyperparams.NUM_CIRCLES):
            base_idx = 3 * circle_idx
            circle_mask = self._create_circle_mask(
                circle_parameters[base_idx],
                circle_parameters[base_idx + 1],
                circle_parameters[base_idx + 2]
            )
            # overlap the masks by taking the maximum value (union)
            mask = torch.maximum(mask, circle_mask)

        log.debug(
            f"Multi-circle mask created with {hyperparams.NUM_CIRCLES} circles each.")
        log.debug(
            f"Mask has shape {mask.shape} with values in range [{mask.min().item()}, {mask.max().item()}]")
        assert mask.sum() > 0, "Generated mask is empty. Check circle parameters and mask creation logic."
        return mask

    def _create_circle_mask(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Create a binary mask of a circle.
        Args:
            x (torch.Tensor): normalized x coordinate of center [0,1]
            y (torch.Tensor): normalized y coordinate of center [0,1]
            r (torch.Tensor): normalized radius [0,1]

        Returns:
            torch.Tensor: (size, size) circle mask
        """
        # scale coordinates
        center_x, center_y, radius = self._normalize_circle_coordinates(
            x, y, r)

        # coordinate grid
        y_coords, x_coords = self.MESHGRID

        # compute distance
        dist_sq = (x_coords - center_x) ** 2 + \
            (y_coords - center_y) ** 2

        # binary mask
        mask = (dist_sq <= radius ** 2).float()

        log.debug(
            f"Circle masks created with centers ({center_x}, {center_y}) and radii {radius}")
        return mask

    def _normalize_circle_coordinates(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize circle parameters from [0,1] to actual pixel coordinates and radius.
        Args:
            x (torch.Tensor): normalized x coordinate of center [0,1]
            y (torch.Tensor): normalized y coordinate of center [0,1]
            r (torch.Tensor): normalized radius [0,1]
        """
        MIN_RADIUS = 10
        size = hyperparams.IMG_SIZE
        center_x = x * size * (1 / 3) + size / 3
        center_y = y * size * (1 / 3) + size / 3
        radius = r * 20 + MIN_RADIUS
        return center_x, center_y, radius
