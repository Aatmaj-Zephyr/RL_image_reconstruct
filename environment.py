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
            torch.arange(hyperparams.IMG_SIZE),
            torch.arange(hyperparams.IMG_SIZE),
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
            self.gt_params)
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
        """Sample random shape parameters for the environment."""
        circles = [torch.rand(3) for _ in range(hyperparams.NUM_CIRCLES)]
        triangles= [torch.rand(6) for _ in range(hyperparams.NUM_TRIANGLES)]
        shapes = circles + triangles
        sampled_params = torch.cat(shapes)
        log.debug(f"Sampled environment parameters: {sampled_params}")
        return sampled_params

    def _create_shape_masks(self, shape_parameters: torch.Tensor) -> torch.Tensor:
        """Create a combined mask for multiple shapes given their parameters.
        Args:
            shape_parameters (torch.Tensor): (3*num_shapes) tensor of shape parameters
        Returns:
            torch.Tensor: (H, W) combined binary mask for all shapes
        """
        log.debug(f"Creating shape masks from parameters: {shape_parameters}")
        assert shape_parameters.shape[0] == hyperparams.NUM_CIRCLES * 3 + hyperparams.NUM_TRIANGLES * 6, f"Expected shape parameters of length \
            {hyperparams.NUM_CIRCLES * 3 + hyperparams.NUM_TRIANGLES * 6}, got {shape_parameters.shape[0]}"
        circle_params = shape_parameters[:3*hyperparams.NUM_CIRCLES]
        triangle_params = shape_parameters[3*hyperparams.NUM_CIRCLES:3*hyperparams.NUM_CIRCLES+6*hyperparams.NUM_TRIANGLES]
        log.debug(f"Separated circle parameters: {circle_params}, triangle parameters: {triangle_params}")

        size = hyperparams.IMG_SIZE
        mask = torch.zeros((size, size))

        for circle_idx in range(hyperparams.NUM_CIRCLES):
            base_idx = 3 * circle_idx
            circle_mask = self._create_circle_mask(
                circle_params[base_idx],
                circle_params[base_idx + 1],
                circle_params[base_idx + 2]
            )
            # overlap the masks by taking the maximum value (union)
            mask = torch.maximum(mask, circle_mask)
        for triangle_idx in range(hyperparams.NUM_TRIANGLES):
            base_idx = 6 * triangle_idx
            triangle_mask = self._create_triangle_mask(triangle_params[base_idx:base_idx+6])
            mask = torch.maximum(mask, triangle_mask)

        log.debug(
            f"Mask has shape {mask.shape} with values in range [{mask.min().item()}, {mask.max().item()}]")
        assert mask.sum() > 0, "Generated mask is empty. Check circle parameters and mask creation logic."
        return mask

    def _create_triangle_mask(self, triangle_params:torch.Tensor) -> torch.Tensor:
        """Create a binary mask of a triangle.
        Args:
            triangle_params torch.Tensor:  List of vertex coordinate tensors [(x1, y1), (x2, y2), (x3, y3)]
        Returns:
            torch.Tensor: (size, size) triangle mask
        """
        # scale coordinates
        vertices = self._normalize_triangle_coordinates(triangle_params)

        # coordinate grid
        y_coords, x_coords = self.MESHGRID

        # Compute edge vectors
        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = torch.stack((x_coords - vertices[0, 0], y_coords - vertices[0, 1]), dim=-1)
        # Compute dot products
        dot00 = (v0 * v0).sum(dim=-1)
        dot01 = (v0 * v1).sum(dim=-1)
        dot02 = (v0 * v2).sum(dim=-1)
        dot11 = (v1 * v1).sum(dim=-1)
        dot12 = (v1 * v2).sum(dim=-1)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        # Create binary mask where u >= 0, v >= 0, and u + v < 1
        mask = ((u >= 0) & (v >= 0) & (u + v < 1)).float()


        log.debug(
            f"Triangle mask created with vertices {vertices} and shape {mask.shape}")
        return mask

    def _normalize_triangle_coordinates(self,triangle_params:torch.Tensor) -> torch.Tensor:
        """Normalize triangle vertex coordinates from [0,1] to actual pixel coordinates.
        Args:
            triangle_params torch.Tensor: Tensor of vertex coordinate tensors [(x1, y1), (x2, y2), (x3, y3)]
        Returns:
            torch.Tensor: Tensor of vertex coordinate tensors [(x1, y1), (x2, y2), (x3, y3)]
        """
        log.debug(f"Normalizing triangle parameters: {triangle_params}")
        size = hyperparams.IMG_SIZE
        extent = size * (1 / 3)
        x1, y1 = triangle_params[0], triangle_params[1]
        x2, y2 = triangle_params[2], triangle_params[3]
        x3, y3 = triangle_params[4], triangle_params[5]

        vertices = torch.tensor([
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ])

        # compute centroid
        centroid = vertices.mean(dim=0)

        scale = hyperparams.TRIANGLE_SCALE  # increase to enlarge triangle

        # push vertices away from centroid
        vertices = centroid + scale * (vertices - centroid)

        # map to canvas
        vertices = torch.stack([
            extent + vertices[:,0] * extent,
            extent + vertices[:,1] * extent
        ], dim=1)

        log.debug(f"Normalized triangle vertices: {vertices}")
        return vertices

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
