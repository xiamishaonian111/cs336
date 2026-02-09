"""Lightweight experiment logging to JSON-lines files."""

import json
import os
import time


class ExperimentLogger:
    """Logs metrics to a JSONL file, one JSON object per line.

    Each logged entry includes the step number and wallclock time (seconds
    since the experiment started). The file can be loaded back with
    ``ExperimentLogger.load()`` for plotting loss curves.

    Usage::

        logger = ExperimentLogger("experiments/run_01")
        logger.start(config={"lr": 1e-3, "batch_size": 32})
        logger.log(step=0, train_loss=4.2, lr=1e-3)
        logger.log(step=100, val_loss=3.8)
        logger.finish()

        # Later, to plot:
        entries = ExperimentLogger.load("experiments/run_01/metrics.jsonl")
    """

    def __init__(self, log_dir: str, overwrite: bool = False) -> None:
        """
        Args:
            log_dir: Directory to write log files into (created on start).
            overwrite: If True, overwrite existing logs. If False, append.
        """
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self._file = None
        self._start_time = None
        self._overwrite = overwrite

    def start(self, config: dict) -> None:
        """Create log directory, save config, open log file, start wallclock.

        Args:
            config: Dict of hyperparameters / run config. Saved to
                    ``config.json`` in log_dir.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        self._file = open(self.metrics_path, "w" if self._overwrite else "a")
        self._start_time = time.time()

        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def log(self, step: int, **metrics: float) -> None:
        """Append one entry with step, wallclock time, and arbitrary metrics."""
        entry = {
            "step": step,
            "wallclock": time.time() - self._start_time,
            **metrics,
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def finish(self) -> None:
        """Close the log file."""
        self._file.close()

    @staticmethod
    def load(path: str) -> list[dict]:
        """Load a metrics.jsonl file and return a list of dicts."""
        entries = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
