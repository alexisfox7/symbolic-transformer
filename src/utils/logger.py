"""Unified logger for the entire project."""
from accelerate import PartialState
from accelerate.logging import get_logger

_ = PartialState()

logger = get_logger("log", log_level="DEBUG")

__all__ = ['logger']