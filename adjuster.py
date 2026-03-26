import re

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cma

from model import REINFORCE
from environment import ShapeDrawEnv
from helpers.hyperparams import hyperparams
import os
import argparse
from helpers.config import config, load_config
from helpers.logger import log, setup_logger
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


def render(gt_mask, pred_mask, iou):
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
    plt.show()


# -----------------------------
# CMA-ES Optimization
# -----------------------------
def refine_with_cmaes(env, gt_mask, init_params, device, min_steps,
                     max_steps, sigma, popsize):
    """
    CMA-ES optimization loop
    """

    x0 = init_params.detach().cpu().numpy()

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma,
        {'popsize': popsize}
    )


    step = 0
    while True:
        step+=1
        solutions = es.ask()
        rewards = []

        for s in solutions:
            s_tensor = torch.tensor(s, dtype=torch.float32)

            pred_mask = env.create_shape_masks(s_tensor).to(device)
            iou = compute_iou(gt_mask, pred_mask)

            rewards.append(-iou)  # minimize

        es.tell(solutions, rewards)
        if step>=min_steps and -min(rewards) > 0.99:  # early stop if IoU is very good
            print(f"Early stop at step {step} with IoU: {-min(rewards):.4f}")
            break
        if step>=max_steps:
            break
        if step % 5 == 0:
            print(f"[CMA step {step}] best IoU: {-min(rewards):.4f}")
        

    best = torch.tensor(
        es.result.xbest,
        dtype=torch.float32
    ).to(device)

    return best


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
    render(gt_mask.cpu(), init_mask.cpu(), init_iou)

    # -----------------------------
    # Step 2: CMA-ES refinement
    # -----------------------------
    refined_params = refine_with_cmaes(
        env,
        gt_mask,
        init_params,
        device,
        min_steps = 100,    # adjust
        max_steps=1000,     # adjust
        sigma=0.1,    # adjust
        popsize=16    # adjust
    )

    # -----------------------------
    # Step 3: Final result
    # -----------------------------
    final_mask = env.create_shape_masks(refined_params.cpu()).to(device)
    final_iou = compute_iou(gt_mask, final_mask)

    print(f"Final IoU after CMA-ES: {final_iou:.4f}")

    render(gt_mask.cpu(), final_mask.cpu(), final_iou)


if __name__ == "__main__":
    main()