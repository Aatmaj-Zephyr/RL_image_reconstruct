"""Training script for ML model."""
import argparse
import torch

from helpers.config import config, load_config
from helpers.logger import log, setup_logger
from helpers.telemetry_writer import telemetry_writer

from trainer import train

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Run mode (debug/prod)",
                    choices=["debug", "prod"], default="debug")
args = parser.parse_args()
load_config(args.mode)


setup_logger()
telemetry_writer.setup_writer(
    fieldnames=["epoch", "train_loss", "val_loss"]  # modify this as needed
)


if __name__ == "__main__":
    if config.IS_DEBUG_TORCH:
        # Note: If using DDP with torch.multiprocessing.spawn, anomaly detection must
        # be enabled inside each worker process, not only in the main file.
        torch.autograd.set_detect_anomaly(True)
        torch.set_warn_always(True)

    log.info(
        f"Starting training with run_id: <red>{config.runtime.RUN_ID}</red> at time: {config.runtime.START_TIME}")
    train()
