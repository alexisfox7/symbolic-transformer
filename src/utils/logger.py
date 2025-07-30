"""Unified logger for the entire project."""
from accelerate import Accelerator

accelerator = Accelerator()

class AcceleratorPrintLogger:
    def __init__(self, accelerator):
        self.accelerator = accelerator
    
    def info(self, msg):
        self.accelerator.print(f"{msg}")
    
    def debug(self, msg):
        self.accelerator.print(f"[DEBUG] {msg}")
    
    def warning(self, msg):
        self.accelerator.print(f"[WARNING] {msg}")
    
    def error(self, msg):
        self.accelerator.print(f"[ERROR] {msg}")

logger = AcceleratorPrintLogger(accelerator)

__all__ = ['logger']