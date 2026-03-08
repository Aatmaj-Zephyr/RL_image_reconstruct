"""Training code goes here."""

import os
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.multiprocessing as mp
from torch import optim
from helpers.config import config
from helpers.hyperparams import hyperparams
from helpers.logger import log
from helpers.telemetry_writer import telemetry_writer
from model import REINFORCE

MESHGRID = torch.meshgrid(
    torch.arange(hyperparams.IMG_SIZE, device=hyperparams.DEVICE),
    torch.arange(hyperparams.IMG_SIZE, device=hyperparams.DEVICE),
    indexing="ij",
)  # meshgrid for geometry calculations


def create_shape_masks(circle_parameters: torch.Tensor) -> torch.Tensor:
    """Create a combined mask for multiple circles given their parameters.
    Args:
        circle_parameters (torch.Tensor): (B, 3*num_circles) tensor of circle parameters
    Returns:
        torch.Tensor: (B, H, W) combined binary mask for all circles
    """
    size = hyperparams.IMG_SIZE
    batch_size = circle_parameters.shape[0]
    mask = torch.zeros((batch_size, size, size),
                       device=circle_parameters.device)

    for circle_idx in range(hyperparams.NUM_CIRCLES):
        base_idx = 3 * circle_idx
        circle_mask = create_circle_mask(
            circle_parameters[:, base_idx],
            circle_parameters[:, base_idx + 1],
            circle_parameters[:, base_idx + 2]
        )
        # overlap the masks by taking the maximum value (union)
        mask = torch.maximum(mask, circle_mask)

    log.debug(
        f"Multi-circle mask created for batch size {batch_size} with {hyperparams.NUM_CIRCLES} circles each.")
    log.debug(
        f"Mask has shape {mask.shape} with values in range [{mask.min().item()}, {mask.max().item()}]")
    assert mask.sum() > 0, "Generated mask is empty. Check circle parameters and mask creation logic."
    return mask


def create_circle_mask(x: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Create a binary mask of a circle.
    Args:
        x (torch.Tensor): normalized x coordinate of center [0,1]
        y (torch.Tensor): normalized y coordinate of center [0,1]
        r (torch.Tensor): normalized radius [0,1]

    Returns:
        torch.Tensor: (size, size) circle mask
    """

    # scale coordinates
    center_x, center_y, radius = normalize_circle_coordinates(x, y, r)

    # coordinate grid
    y_coords, x_coords = MESHGRID

    # compute distance
    dist_sq = (x_coords - center_x[:, None, None]) ** 2 + \
        (y_coords - center_y[:, None, None]) ** 2

    # binary mask
    mask = (dist_sq <= radius[:, None, None] ** 2).float()

    log.debug(
        f"Circle masks created with centers ({center_x.detach().cpu().numpy()}, {center_y.detach().cpu().numpy()}) and radii {radius.detach().cpu().numpy()}")
    return mask


def normalize_circle_coordinates(x: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize circle parameters from [0,1] to actual pixel coordinates and radius.
    Args:
        x (torch.Tensor): normalized x coordinate of center [0,1]
        y (torch.Tensor): normalized y coordinate of center [0,1]
        r (torch.Tensor): normalized radius [0,1]
    """
    MIN_RADIUS = 10
    size = hyperparams.IMG_SIZE
    center_x = x * size * (3 / 5) + size / 5
    center_y = y * size * (3 / 5) + size / 5
    radius = r * 40 + MIN_RADIUS
    return center_x, center_y, radius


def compute_batched_reward(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Compute IoU reward for a batch of predicted and ground truth masks.
    Args:
        pred_mask (torch.Tensor): (B, H, W) predicted binary masks
        gt_mask (torch.Tensor): (B, H, W) ground truth binary masks
    Returns:
        torch.Tensor: (B,) reward values for each sample in the batch
    """
    assert pred_mask.shape == gt_mask.shape, "Predicted and GT masks must have the same shape"

    intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
    union = pred_mask.sum(dim=(1, 2)) + \
        gt_mask.sum(dim=(1, 2)) - intersection + 1e-6

    reward = intersection / union
    log.debug(f"Reward computed over batch: {reward}")
    return reward


def rollout_and_render(episode_id: int, gt_params: torch.Tensor, policy_network: torch.nn.Module) -> None:
    """
    Run deterministic rollout.
    Note (argmax = mean prediction)
    and renders:
        - GT circle in red (alpha 0.5)
        - Predicted circle in green (alpha 0.5)
        - Overlap in yellow (alpha 1.0)
    Args:
    episode_id (int): current episode index
    gt_params (torch.Tensor): ground truth circle parameters (3*num_circles,)
    policy_network (torch.nn.Module): The trained policy network to generate predictions

    """

    gt_mask, pred_mask = rollout(gt_params, policy_network)

    render(episode_id, gt_mask, pred_mask)


def render(episode_id: int, gt_mask: torch.Tensor, pred_mask: torch.Tensor, show: bool = False, ) -> None:
    """
        Visualize the overlap between ground truth and predicted masks.

        This function creates an RGB visualization where:
        - Red channel represents the ground truth mask
        - Green channel represents the predicted mask
        - Yellow regions indicate overlap between the two masks

        The visualization can optionally be saved to disk and/or displayed.

        Args:
            episode_id (int):
                Identifier for the current episode, used for labeling the plot.

            gt_mask (torch.Tensor):
                Ground truth binary mask of shape (H, W). Non-zero values indicate
                pixels belonging to the true shape.

            pred_mask (torch.Tensor):
                Predicted binary mask of shape (H, W). Non-zero values indicate
                pixels belonging to the predicted shape.


            show (bool, optional):
                If True, the visualization will be displayed using matplotlib.
                Defaults to False.

        Returns:
            None
    """
    gt_mask = gt_mask.detach().cpu()
    pred_mask = pred_mask.detach().cpu()

    log.debug(
        f"Rendering episode {episode_id} with GT mask {gt_mask} and Pred mask {pred_mask}")
    log.debug(
        f"gt_mask shape: {gt_mask.shape}, pred_mask shape: {pred_mask.shape}")

    # we will display only the first image of the batch
    overlap = gt_mask[0] * pred_mask[0]

    # Compute reward (IoU) for display
    reward = compute_batched_reward(pred_mask.unsqueeze(0), gt_mask).item()
    # Create RGB canvas
    H, W = gt_mask[0].shape
    canvas = torch.zeros((H, W, 3))

    # Red = GT
    canvas[..., 0] = gt_mask[0]

    # Green = Prediction
    canvas[..., 1] = pred_mask

    # Yellow overlap
    canvas[..., 0][overlap > 0] = 1
    canvas[..., 1][overlap > 0] = 1
    canvas = torch.clamp(canvas, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(canvas)
    plt.title(f"Episode {episode_id} | IoU: {reward:.3f}")
    plt.axis("off")
    fig_name = f"{config.runtime.RUN_ID}_episode_{episode_id:07d}.png"
    os.makedirs(config.IMG_SAVE_PATH, exist_ok=True)
    save_path = os.path.join(config.IMG_SAVE_PATH, fig_name)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"Episode {episode_id} | IoU: {reward:.3f}")
    if show:
        plt.show()
    plt.close()


def rollout(gt_params: torch.Tensor, policy_network: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a deterministic rollout of the policy network.

    Args:
        gt_params: Tensor of shape (batch_size, num_params) containing the ground truth parameters for the circles.
        policy_network: The trained policy network to generate predictions.
    Returns:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask from the policy network
    """

    gt_params = gt_params.to(hyperparams.DEVICE)
    gt_params = gt_params.unsqueeze(0)  # add batch dimension

    log.debug(
        f"Performing rollout with GT parameters: {gt_params.cpu().numpy()}")

    # Create ground truth mask
    gt_mask = create_shape_masks(gt_params)

    # Prepare image input (3 channel)
    image = gt_mask.unsqueeze(0)
    policy_network.eval()  # set to eval mode to disable learning

    # Deterministic action (argmax for Gaussian = mean)
    pred_params = policy_network(image)
    policy_network.train()  # set to train mode to enable learning

    log.debug(
        f"Obtained predicted params: {pred_params.detach().cpu().numpy()}")

    pred_mask = create_shape_masks(pred_params)[0]

    return gt_mask, pred_mask


def save_model(episode_index: int, policy_network: torch.nn.Module) -> None:
    """Save the model.
    Args:
        episode_index (int): current episode index
        policy_network (torch.nn.Module): The policy network to save
    """
    save_path = os.path.join(config.MODEL_SAVE_PATH,
                             f"model_{config.runtime.RUN_ID}_{episode_index:07d}.pth")
    torch.save(policy_network.state_dict(), save_path)


def sample_environment_cpu() -> torch.Tensor:
    """Sample random circle parameters for the environment."""
    circles = [torch.rand(3) for _ in range(hyperparams.NUM_CIRCLES)]
    return torch.cat(circles)


def worker_fn(queue: mp.Queue) -> None:
    """Worker function to sample environment parameters and put them in the queue.
    Args:        queue (mp.Queue): multiprocessing queue to put sampled parameters
    """
    while True:
        gt_params = sample_environment_cpu()
        queue.put(gt_params)


def start_workers(num_workers: int) -> tuple[mp.Queue, list]:
    """Start worker processes for environment sampling.
    Args:
        num_workers (int): number of worker processes to start
    Returns:
        mp.Queue: multiprocessing queue shared among workers for sampled parameters
        list: list of worker processes
    """
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=100)

    workers = []
    for _ in range(num_workers):
        p = context.Process(target=worker_fn, args=(queue,))
        p.daemon = True
        p.start()
        workers.append(p)

    return queue, workers


# trunk-ignore(pylint/R0915)
# trunk-ignore(pylint/R0912)
# trunk-ignore(pylint/R0914)
def train() -> None:
    """Train the model."""
    if config.IS_DEBUG_TORCH:
        # Note: If using DDP with torch.multiprocessing.spawn, anomaly detection must
        # be enabled inside each worker process, not only in the main file.
        torch.autograd.set_detect_anomaly(True)
        torch.set_warn_always(True)
    log.debug("In the training function")

    print("Running training loop MNIST CIRCLES")
    T = hyperparams.TEMP_START
    mp.set_start_method("spawn", force=True)

    policy_network = REINFORCE().to(hyperparams.DEVICE)

    optimizer = optim.AdamW(
        policy_network.parameters(),
        lr=hyperparams.LEARNING_RATE,
        weight_decay=hyperparams.WEIGHT_DECAY_ADAM)

    queue, _ = start_workers(hyperparams.BATCH_SIZE)
    moving_avgs_rewards_list = []

    for episode in range(hyperparams.NUM_EPISODES):

        gt_params = torch.stack([queue.get()
                                for _ in range(hyperparams.BATCH_SIZE)])

        gt_mask = create_shape_masks(gt_params.to(hyperparams.DEVICE))

        images = gt_mask.unsqueeze(1)
        mean = policy_network(images)
        std = torch.full_like(mean, T)
        dist = D.Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=1)
        action_xy = torch.clip(raw_action, 0, 1)

        action = torch.cat([action_xy], dim=1)
        pred_mask = create_shape_masks(action)

        rewards = compute_batched_reward(pred_mask, gt_mask)
        moving_avgs_rewards_list.append(rewards.mean().item())
        if len(moving_avgs_rewards_list) > 100:
            moving_avgs_rewards_list.pop(0)
        baseline = rewards.mean()
        advantage = rewards - baseline

        loss = -(log_prob * advantage.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy_network.parameters(), max_norm=1.0)

        optimizer.step()

        if episode % hyperparams.TEMP_UPDATE_INTERVAL == 0:
            T = max(T * hyperparams.TEMP_DECAY, hyperparams.TEMP_MIN)
        if episode % config.TELEMETRY_INTERVAL == 0:
            telemetry_writer.log(
                episode=episode,
                moving_avg_reward=(
                    sum(moving_avgs_rewards_list) / len(moving_avgs_rewards_list)), temperature=T
            )
        if episode % config.ROLLOUT_INTERVAL == 0:
            rollout_and_render(episode, gt_params[0], policy_network)
        if episode % config.CHECKPOINT_INTERVAL == 0:
            save_model(episode_index=episode, policy_network=policy_network)
