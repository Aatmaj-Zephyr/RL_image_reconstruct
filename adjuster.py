import re
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cma
import gymnasium as gym
from model import REINFORCE
from environment import ShapeDrawEnv
from helpers.hyperparams import hyperparams
import os
import argparse
from helpers.config import config, load_config
from helpers.logger import log, setup_logger, setup_worker_logger
from helpers.telemetry_writer import telemetry_writer

MODEL_PATH = "/Users/aatmaj/RL_image_reconstruct/models/model_well-hound_0700000.pth"
IMAGE_PATH = "/Users/aatmaj/RL_image_reconstruct/custom_tests/circle_triangle_rectangle1.png"


# -----------------------------
# Utils
# -----------------------------
def load_mask(path):
    img = Image.open(path).convert("L").resize(
        (hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)
    )
    arr = np.array(img) / 255.0
    mask = (arr > 0.5).astype(np.float32)
    return torch.tensor(mask, dtype=torch.float32)


def compute_iou(gt_mask, pred_mask):
    intersection = (gt_mask * pred_mask).sum()
    union = gt_mask.sum() + pred_mask.sum() - intersection + 1e-6
    return (intersection / union).item()


def render(gt_mask, pred_mask, iou,show = True):
    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()
    overlap = gt * pred

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Ground Truth
    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Overlap
    canvas = np.zeros((*gt.shape, 3))
    canvas[..., 0] = gt
    canvas[..., 1] = pred

    mask = overlap > 0
    canvas[..., 0][mask] = 1
    canvas[..., 1][mask] = 1

    axes[2].imshow(canvas)
    axes[2].set_title(f"Overlap (IoU={iou:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    if show:
        plt.show()

    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())

    return frame


def make_env() -> gym.Env:
    """Return a new environment. (This is a helper function to return environment object.)"""
    run_id = os.environ.get("RL_RUN_ID")
    if run_id:
        level = os.environ.get("RL_LOG_LEVEL", "INFO")
        setup_worker_logger(run_id=run_id, level=level)
    return ShapeDrawEnv()
# -----------------------------
# CMA-ES Optimization
# -----------------------------
def refine_with_cmaes(gt_mask, init_params, device):
    """
    CMA-ES optimization loop
    """
    envs = gym.vector.AsyncVectorEnv(
        [make_env for _ in range(hyperparams.POP_SIZE)])
    envs.reset()
 
    intermediate_bests = []  # to store intermediate best solutions for video 
    envs.call("set_target", gt_mask.cpu().numpy())
    x0 = init_params.detach().cpu().numpy()

    es = cma.CMAEvolutionStrategy(
        x0,
        hyperparams.CMA_SIGMA,  # adjust
        {'popsize': hyperparams.POP_SIZE}
    )

    step = 0
    while True:
        step+=1
        solutions = es.ask()
        rewards = []

        # Convert to numpy batch
        actions = np.stack(solutions).astype(np.float32)

        # Step all envs in parallel
        _, rewards, _, _, _ = envs.step(actions)

        rewards = -rewards  # CMA minimizes

        es.tell(solutions, rewards)
        best_iou = -min(rewards)

        if step % 5 == 0:
            print(f"[CMA step {step}] best IoU: {best_iou:.4f}")
            intermediate_bests.append((step, best_iou, es.result.xbest.copy()))
        if step >= hyperparams.MIN_CMA_STEPS and best_iou > hyperparams.CMA_REWARD_SATISFACTION_THRESHOLD:
            print(f"Early stop at step {step} with IoU: {best_iou:.4f}")
            break

        if step >= hyperparams.MAX_CMA_STEPS:
            break

    best = torch.tensor(
        es.result.xbest,
        dtype=torch.float32
    ).to(device)

    return best,intermediate_bests

def video_save(frames, path, fps):
    if not frames:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()

def video_maker(intermediate_bests, gt_mask, env, device):
    """Make video from intermediate best solutions."""
    frames = []
    for step, iou, params in intermediate_bests:
        pred_mask = env.create_shape_masks(params).to(device)
        frame = render(gt_mask.cpu(), pred_mask.cpu(), iou, show=False)
        frames.append(frame)

    path = "cmaes_refinement.mp4"
    fps = 5
    video_save(frames, path, fps)
# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Run mode (debug/prod)",
                        choices=["debug", "prod"], default="debug")
    args = parser.parse_args()
    load_config(args.mode)
    # Pass logging context to multiprocessing workers.
    os.environ["RL_RUN_ID"] = config.runtime.RUN_ID
    os.environ["RL_LOG_LEVEL"] = "DEBUG" if config.IS_DEBUG else "INFO"

    setup_logger()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Load model
    model = REINFORCE().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load environment
    env = ShapeDrawEnv()

    # Load GT
    gt_mask = load_mask(IMAGE_PATH).to(device)

    # Prepare input (same as training)
    image = gt_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    print("Input shape:", image.shape)

    # -----------------------------
    # Step 1: Initial prediction
    # -----------------------------
    with torch.no_grad():
        init_params = model(image)[0]

    print("Initial params:", init_params.cpu().numpy())

    init_mask = env.create_shape_masks(init_params.cpu()).to(device)
    init_iou = compute_iou(gt_mask, init_mask)

    print(f"Initial IoU: {init_iou:.4f}")
    _= render(gt_mask.cpu(), init_mask.cpu(), init_iou)

    # -----------------------------
    # Step 2: CMA-ES refinement
    # -----------------------------
    refined_params, intermediate_bests = refine_with_cmaes(
        gt_mask,
        init_params,
        device
    )

    # -----------------------------
    # Step 3: Final result
    # -----------------------------
    final_mask = env.create_shape_masks(refined_params.cpu()).to(device)
    final_iou = compute_iou(gt_mask, final_mask)

    print(f"Final IoU after CMA-ES: {final_iou:.4f}")

    _=render(gt_mask.cpu(), final_mask.cpu(), final_iou)
    
    video_maker(intermediate_bests, gt_mask, env, device)


if __name__ == "__main__":
    main()