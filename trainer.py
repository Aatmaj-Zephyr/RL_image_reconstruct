"""Training code goes here."""

import time


from helpers.hyperparams import hyperparams
from helpers.telemetry_writer import telemetry_writer
from helpers.logger import log
from helpers.config import config
# from model import *


def train() -> None:
    """Train the model."""
    log.debug('In the training function')
    for epoch in range(hyperparams.NUM_EPOCHS):
        if epoch % config.TELEMETRY_INTERVAL == 0:
            telemetry_writer.log(epoch=epoch, train_loss=0.5, val_loss=0.6)
            execution_time = time.time() - config.runtime.START_TIME
            execution_time_formatted = time.strftime(
                "%H:%M:%S", time.gmtime(execution_time))
            log.info(
                f"Epoch: {epoch}, Elapsed time: {execution_time_formatted}")
