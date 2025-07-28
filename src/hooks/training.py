from typing import Any, Dict
from accelerate.logging import get_logger # type: ignore
from .base import TrainingHook
import os
import json  
import math
from datetime import datetime
logger = get_logger(__name__)

class ConsoleLogHook(TrainingHook):
    def __init__(self, log_every_n_batches: int):
        super().__init__("console_log")
        self.log_every_n_batches = log_every_n_batches

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        epochs = state.get('num_epochs', '?')
        model_params = state.get('model_params', '?')
        logger.info(f"Training started: {epochs} epochs, {model_params:,} parameters")
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        logger.info("Training ended!")
    
    def on_epoch_begin(self, epoch: int, state: Dict[str, Any]) -> None:
        epoch_idx = state.get("current_epoch")
        total = state.get("num_epochs")
        logger.info(f"Epoch {epoch_idx}/{total}")

    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        epoch_idx = state.get("current_epoch")
        avg_loss = state.get("avg_epoch_loss", 0.0)
        epoch_duration = state.get("epoch_duration", 0.0)
        logger.info(f"Epoch {epoch_idx} completed: avg_loss={avg_loss:.4f}, duration={epoch_duration:.2f}s")
    
    def on_batch_begin(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        current_epoch = state.get("current_epoch")
        total_epochs = state.get("num_epochs")

        if batch_idx % self.log_every_n_batches == 0:
            logger.info(f"Epoch {current_epoch}/{total_epochs}, Batch {batch_idx}")

    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        if batch_idx % self.log_every_n_batches == 0:
            current_epoch = state.get("current_epoch")
            total_epochs = state.get("num_epochs")
            logger.info(f"Epoch {current_epoch}/{total_epochs}, Batch {batch_idx} completed: loss={loss:.4f}")

class JSONLogHook(TrainingHook):
    def __init__(self, output_dir: str, log_every_n_batches: int = 100):
        super().__init__("json_log")
        self.log_every_n_batches = log_every_n_batches
        
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"training_{timestamp}.jsonl"
        self.log_file = os.path.join(output_dir, filename)

        initial_data = {"event": "train_start", "log_file": self.log_file}
        self._write(initial_data)

    def _calc_perplexity(self, loss):
        return math.exp(loss)
    
    def _write(self, data):
        data["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')
    
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        config = {k: v for k, v in state.items() 
                 if not callable(v) and k not in ['model', 'optimizer', 'dataloader', 'accelerator']}
        self._write({"event": "train_begin", "config": config})
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        
        # training metrics
        avg_loss = state.get('avg_loss', state.get('loss'))
        epoch_data = {
            "event": "epoch_end",
            "epoch": epoch,
            "loss": avg_loss,
            "duration": state.get('epoch_duration')
        }

        epoch_data["perplexity"] = self._calc_perplexity(avg_loss)
        
        # add validation metrics if available
        val_loss = state.get('val_loss')
        val_perplexity = state.get('val_perplexity')
        if val_loss is not None:
            epoch_data["val_loss"] = val_loss
            epoch_data["val_perplexity"] = val_perplexity or self._calc_perplexity(val_loss)
        
        self._write(epoch_data)
    
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        batch_idx = state.get("current_batch_idx")
        loss = state.get("latest_batch_loss")

        if batch_idx % self.log_every_n_batches == 0:
            if not state.get('is_main_process', True):
                return
            
            batch_data = {
                "event": "batch",
                "epoch": state.get('current_epoch', 0),
                "batch": batch_idx,
                "loss": loss,
                "perplexity": self._calc_perplexity(loss)
            }
            self._write(batch_data)

