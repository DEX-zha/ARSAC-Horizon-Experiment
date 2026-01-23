"""Progress bar utilities for horizon experiments."""

import time


class ProgressBar:
    """Simple progress bar with ETA."""

    def __init__(self, total, label="progress"):
        self.total = max(1, int(total))
        self.label = label
        self.start = time.time()
        self.current = 0

    def update(self, step=1, extra=""):
        """Advances the progress bar and prints the current status."""
        self.current = min(self.total, self.current + step)
        frac = self.current / self.total
        width = 24
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - self.start
        rate = self.current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.current) / rate if rate > 0 else 0.0
        msg = (
            f"\r{self.label} [{bar}] {self.current}/{self.total} "
            f"{frac*100:5.1f}% ETA {eta:5.1f}s {extra}"
        )
        print(msg, end="", flush=True)

    def close(self):
        """Ends the progress bar line."""
        print()
