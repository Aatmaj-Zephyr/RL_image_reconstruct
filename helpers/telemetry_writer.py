"""File to setup a simple telemetry writer that logs metrics to a CSV file. No need to modify this file."""
from typing import Any, TextIO
import csv
import os
import time

from helpers.config import config


class TelemetryWriter:
    """Class to write telemetry to some file."""

    def __init__(self) -> None:
        """Declare the attributes of the class."""
        self.writer: csv.DictWriter
        self.file: TextIO
        self.filepath: str

    def setup_writer(self, fieldnames: list[str], directory: str = "./telemetry_logs") -> None:
        """Set the writer attributes.

        Args:
            fieldnames (list): List of metric names to log (e.g. ["epoch", "train_loss", "val_loss"])
            directory (str, optional): Directory to store the log files. Defaults to "./telemetry_logs".
        """
        assert config.runtime.RUN_ID is not None, "RUN_ID must be set in runtime before setting up telemetry writer."
        os.makedirs(directory, exist_ok=True)

        self.filepath = os.path.join(directory, f"{config.runtime.RUN_ID}.csv")
        file_exists = os.path.isfile(self.filepath)

        # In our case, we're intentionally keeping the file open for repeated writes — which is actually more efficient
        # than opening and closing it every time we log. This is why "with" keyword is not used. The current design is good for performance.
        # trunk-ignore(pylint/R1732)
        self.file = open(self.filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["timestamp"] + fieldnames
        )

        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log(self, **metrics: Any) -> None:
        """Log the metrics.
        """
        assert self.writer is not None, "TelemetryWriter not initialized. Call setup_writer first."
        metrics["timestamp"] = time.time()
        self.writer.writerow(metrics)
        self.file.flush()


telemetry_writer = TelemetryWriter()
