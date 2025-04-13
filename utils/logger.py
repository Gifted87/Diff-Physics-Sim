import os
import numpy as np
from collections import defaultdict
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = defaultdict(list)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            print("Warning: tensorboard not found. Logging to console/files only.")
        print(f"Logging to: {log_dir}")

    def log_scalar(self, tag, value, step):
        self.scalars[tag].append(value)
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def dump(self, step):
        # Optionally write scalars to a file (e.g., csv)
        # For now, mainly relies on TensorBoard
        if self.writer:
            self.writer.flush()
        # Clear internal buffer? Or keep history? Keep for now.
        # self.scalars.clear()

    def close(self):
        if self.writer:
            self.writer.close()