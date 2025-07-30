"""Unified logger for the entire project."""
from accelerate import PartialState
from accelerate.logging import get_logger

_ = PartialState()

logger = get_logger("log")

__all__ = ['logger']