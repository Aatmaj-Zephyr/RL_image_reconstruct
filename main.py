"""Training script for ML model."""
import argparse
import os

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

    # Pass logging context to multiprocessing workers.
    os.environ["RL_RUN_ID"] = config.runtime.RUN_ID
    os.environ["RL_LOG_LEVEL"] = "DEBUG" if config.IS_DEBUG else "INFO"

    setup_logger()
    telemetry_writer.setup_writer(
        fieldnames=["episode", "mean_reward",
                    "temperature"]  # modify this as needed
    )
    log.info(
        f"Starting training with run_id: <red>{config.runtime.RUN_ID}</red> at time: {config.runtime.START_TIME}")
    train()
