# Hook System Documentation

## Overview

The hook system provides a clean way to extend trainer functionality without modifying core training logic. Inspired by TransformerLens, it allows you to inject custom behavior at specific training events.

---

## Quick Start

```python
from trainers import get_trainer

# Create trainer
trainer = get_trainer("simple", model, dataloader, optimizer, device)

# Add built-in hooks
trainer.add_console_logging(log_every_n_batches=10)
trainer.add_json_logging(log_every_n_batches=100)  
trainer.add_checkpointing(save_every_n_epochs=1)

# Train with hooks
trainer.train()
```

---

## Architecture

```
TrainingHook (base class)
    ↓
HookManager (manages multiple hooks)
    ↓  
BaseTrainer (hook integration)
    ↓
SimpleTrainer / AccelerateTrainer (concrete implementations)
```

---

## Hook Lifecycle Events

```python
trainer.train()
# ↓ calls hooks at these points:

hooks.on_train_begin(state)              # Training starts
    hooks.on_epoch_begin(epoch, state)      # Each epoch starts
        hooks.on_batch_end(batch, loss, state)  # After each batch
    hooks.on_epoch_end(epoch, state)        # Each epoch ends
hooks.on_train_end(state)                # Training complete
```

---

## Creating Custom Hooks

```python
from trainers.hooks import TrainingHook

class CustomHook(TrainingHook):
    def __init__(self):
        super().__init__("my_custom_hook")
        self.losses = []
    
    def on_batch_end(self, batch_idx, loss, state):
        self.losses.append(loss)
        if batch_idx % 100 == 0:
            print(f"Average loss: {sum(self.losses)/len(self.losses):.4f}")

# Use it
trainer.add_hook(CustomHook())
```

---

## Built-in Hooks

| Hook            | Purpose                        | Configuration               |
|-----------------|--------------------------------|-----------------------------|
| `ConsoleLogHook`| Progress logging to console    | `log_every_n_batches=10`    |
| `JSONLogHook`   | Structured logging to JSON     | `log_every_n_batches=100`   |
| `CheckpointHook`| Save model checkpoints         | `save_every_n_epochs=1`     |

---

## Hook Management

```python
# Add hooks
trainer.add_hook(my_hook)
trainer.add_console_logging()

# Manage hooks
trainer.remove_hook("console_log")
trainer.disable_hook("json_log")
trainer.enable_hook("json_log")

# List active hooks
print(trainer.hooks.list_hooks())
```

---

## Trainer State

Hooks receive a state dictionary with training information:

```python
state = {
    'model': model,
    'optimizer': optimizer,
    'device': device,
    'current_epoch': 3,
    'current_batch_idx': 150,
    'latest_loss': 0.245,
    'avg_loss': 0.312,
    'model_params': 125_000_000,
    'status': 'training'  # or 'completed', 'error'
}
```

---

## Implementation Status

✅ **Complete**:
- Hook base classes and manager  
- Built-in hooks (console, JSON, checkpoint)  
- BaseTrainer integration  
- Hook management methods

⚠️ **In Progress**:
- Hook calls in concrete trainers (`SimpleTrainer`, `AccelerateTrainer`)


