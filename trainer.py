"""Training code goes here."""

import os
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.multiprocessing as mp
from torch import optim
import gymnasium as gym

from helpers.config import config
from helpers.hyperparams import hyperparams
from helpers.logger import log, setup_worker_logger
from helpers.telemetry_writer import telemetry_writer
from model import REINFORCE
from environment import ShapeDrawEnv


def rollout_and_render(episode_id: int, envs: gym.vector.AsyncVectorEnv, policy_network: torch.nn.Module) -> torch.Tensor:
    """
    Run deterministic rollout.
    Note (argmax = mean prediction)
    and renders:
        - GT circle in red (alpha 0.5)
        - Predicted circle in green (alpha 0.5)
        - Overlap in yellow (alpha 1.0)
    Args:
    episode_id (int): current episode index
    envs (gym.vector.AsyncVectorEnv): The environment instance to sample GT parameters and create masks
    policy_network (torch.nn.Module): The trained policy network to generate predictions
    Returns:
    reward (torch.Tensor): The computed rewardof the batch
    """

    gt_mask, pred_mask, reward = rollout(envs, policy_network)

    render(episode_id, gt_mask, pred_mask, reward)
    return reward


def rollout(env: gym.vector.AsyncVectorEnv, policy_network: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run a deterministic rollout of the policy network.

    Args:
        envs (gym.vector.AsyncVectorEnv): The environment instance to sample GT parameters and create masks.
        policy_network (torch.nn.Module): The trained policy network to generate predictions.

    Returns:
        gt_mask (torch.Tensor): Ground truth mask
        pred_mask (torch.Tensor): Predicted mask from the policy network
        reward (torch.Tensor): The computed reward (e.g., IoU) for the current prediction
    """

    # Create ground truth mask
    gt_mask, info = env.reset()

    log.info(f"GT_params {info['gt_params']}")
    log.debug(f"Rollout GT mask shape: {gt_mask.shape}")
    gt_mask = torch.tensor(gt_mask, dtype=torch.float32,
                           device=hyperparams.DEVICE)
    image = gt_mask.unsqueeze(1).repeat(1, 3, 1, 1)
    policy_network.eval()  # set to eval mode to disable learning

    # Deterministic action (argmax for Gaussian = mean)
    pred_params = policy_network(image)
    policy_network.train()  # set to train mode to enable learning

    log.info(
        f"Obtained predicted params: {pred_params.detach().cpu().numpy()}")

    pred_mask, reward, _, _, _ = env.step(pred_params.detach().cpu().numpy())
    pred_mask = torch.tensor(
        pred_mask, dtype=torch.float32, device=hyperparams.DEVICE)
    return gt_mask, pred_mask, reward


def render(episode_id: int, gt_masks: torch.Tensor, pred_masks: torch.Tensor, rewards: torch.Tensor, show: bool = False, ) -> None:
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

            gt_masks (torch.Tensor):
                Ground truth binary mask of shape (H, W). Non-zero values indicate
                pixels belonging to the true shape.

            pred_masks (torch.Tensor):
                Predicted binary mask of shape (H, W). Non-zero values indicate
                pixels belonging to the predicted shape.

            reward (torch.Tensor):
                The reward (e.g., IoU) computed for the current prediction, used for
                display in the plot title.

            show (bool, optional):
                If True, the visualization will be displayed using matplotlib.
                Defaults to False.

        Returns:
            None
    """
    gt_mask = gt_masks[0].cpu()
    pred_mask = pred_masks[0].cpu()
    reward = rewards[0].item()  # get the first env's reward
    log.info(f"Episode {episode_id} | IoU: {reward:.3f}")

    # we will display only the first image of the batch
    overlap = gt_mask * pred_mask

    # Create RGB canvas
    H, W = gt_mask.shape
    canvas = torch.zeros((H, W, 3))

    # Red = GT
    canvas[..., 0] = gt_mask
    # Green = Prediction
    canvas[..., 1] = pred_mask

    # Yellow overlap
    canvas[..., 0][overlap > 0] = 1
    canvas[..., 1][overlap > 0] = 1
    canvas = torch.clamp(canvas, 0, 1)

    # show and display
    plt.figure(figsize=(4, 4))
    plt.imshow(canvas)
    plt.title(f"Episode {episode_id} | IoU: {reward:.3f}")
    plt.axis("off")
    fig_name = f"{config.runtime.RUN_ID}_episode_{episode_id:07d}.png"
    os.makedirs(config.IMG_SAVE_PATH, exist_ok=True)
    save_path = os.path.join(config.IMG_SAVE_PATH, fig_name)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close()


def save_model(episode_index: int, policy_network: torch.nn.Module) -> None:
    """Save the model.
    Args:
        episode_index (int): current episode index
        policy_network (torch.nn.Module): The policy network to save
    """
    save_path = os.path.join(config.MODEL_SAVE_PATH,
                             f"model_{config.runtime.RUN_ID}_{episode_index:07d}.pth")
    torch.save(policy_network.state_dict(), save_path)


def make_env() -> gym.Env:
    """Return a new environment. (This is a helper function to return environment object.)"""
    run_id = os.environ.get("RL_RUN_ID")
    if run_id:
        level = os.environ.get("RL_LOG_LEVEL", "INFO")
        setup_worker_logger(run_id=run_id, level=level)
    return ShapeDrawEnv()


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

    # optimizer = optim.AdamW(
    #     policy_network.parameters(),
    #     lr=hyperparams.LEARNING_RATE,
    #     weight_decay=hyperparams.WEIGHT_DECAY_ADAM)
    optimizer = optim.Adam(
        policy_network.parameters(),
        lr=hyperparams.LEARNING_RATE)

    envs = gym.vector.AsyncVectorEnv(
        [make_env for _ in range(hyperparams.NUM_ENVS)])
    log.debug("Made environments")

    for episode in range(hyperparams.NUM_EPISODES):

        gt_mask, _ = envs.reset()  # get GT parameters from vectorized env reset

        log.debug(f"Episode {episode} | GT mask shape: {gt_mask.shape}")

        gt_mask = torch.tensor(
            gt_mask, dtype=torch.float32, device=hyperparams.DEVICE)
        image = gt_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        log.debug(f"Episode {episode} | Image shape: {image.shape}")
        mean = policy_network(image)
        std = torch.full_like(mean, T)
        dist = D.Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=1)
        action = torch.clamp(raw_action, 0, 1)
        _, rewards, _, _, _ = envs.step(action.cpu().numpy())

        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=hyperparams.DEVICE)
        baseline = rewards.mean()
        advantage = rewards - baseline

        loss = -(log_prob * advantage.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     policy_network.parameters(), max_norm=1.0)

        optimizer.step()

        if episode % hyperparams.TEMP_UPDATE_INTERVAL == 0:
            T = max(T * hyperparams.TEMP_DECAY, hyperparams.TEMP_MIN)

        if episode % config.ROLLOUT_INTERVAL == 0:
            reward = rollout_and_render(episode, envs, policy_network)
            telemetry_writer.log(
                episode=episode,
                mean_reward=(reward.mean()), temperature=T
            )
        if episode % config.CHECKPOINT_INTERVAL == 0:
            save_model(episode_index=episode, policy_network=policy_network)
