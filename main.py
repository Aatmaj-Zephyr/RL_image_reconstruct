"""Training script for ML model."""
import argparse
import torch

from helpers.config import config, load_config
from helpers.logger import log, setup_logger
from helpers.telemetry_writer import telemetry_writer

from trainer import train

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Run mode (debug/prod)",
                        choices=["debug", "prod"], default="debug")
    args = parser.parse_args()
    load_config(args.mode)


    setup_logger()
    telemetry_writer.setup_writer(
        fieldnames=["episode", "total_rewards", "moving_avg_reward"]  # modify this as needed
    )
    log.info(
        f"Starting training with run_id: <red>{config.runtime.RUN_ID}</red> at time: {config.runtime.START_TIME}")
    train()
