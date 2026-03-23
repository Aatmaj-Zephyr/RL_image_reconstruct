# evaluate_mickey_proper.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import REINFORCE
from environment import ShapeDrawEnv
from helpers.hyperparams import hyperparams


MODEL_PATH = "/Users/aatmaj/RL_image_reconstruct/models/model_well-hound_0400000.pth"
IMAGE_PATH = "/Users/aatmaj/RL_image_reconstruct/custom_tests/circle_triangle_rectangle5.png"


def load_mask(path):
    img = Image.open(path).convert("L").resize(
        (hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)
    )
    arr = np.array(img) / 255.0

    # IMPORTANT: adjust depending on your image
    mask = (arr > 0.5).astype(np.float32)

    return torch.tensor(mask, dtype=torch.float32)

def render(gt_mask, pred_mask, iou):
    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()
    overlap = gt * pred

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Ground Truth
    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # 2. Prediction
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # 3. Overlap (RGB)
    canvas = np.zeros((*gt.shape, 3))
    canvas[..., 0] = gt        # red
    canvas[..., 1] = pred      # green

    # yellow overlap
    mask = overlap > 0
    canvas[..., 0][mask] = 1
    canvas[..., 1][mask] = 1

    axes[2].imshow(canvas)
    axes[2].set_title(f"Overlap (IoU={iou:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def main():
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

    # Load GT mask from image
    gt_mask = load_mask(IMAGE_PATH).to(device)

    

    # Prepare input (same as training)
    image = gt_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    print("Input shape:", image.shape)

    # Predict parameters
    with torch.no_grad():
        pred_params = model(image)[0]  # remove batch dim

    print("Pred params:", pred_params.cpu().numpy())

    # Generate predicted mask USING ENV (CRITICAL STEP)
    pred_mask = env.create_shape_masks(pred_params.cpu()).to(device)

    # Ensure same shape
    assert gt_mask.shape == pred_mask.shape

    # Compute IoU
    intersection = (gt_mask * pred_mask).sum()
    union = gt_mask.sum() + pred_mask.sum() - intersection + 1e-6
    iou = (intersection / union).item()

    print(f"IoU: {iou:.4f}")

    # Render
    render(gt_mask.cpu(), pred_mask.cpu(),iou)


if __name__ == "__main__":
    main()