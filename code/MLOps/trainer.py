"""
Training loop and optimization utilities.

Physics perspective on optimization:
- Gradient descent ≈ steepest descent in loss landscape
- SGD with momentum ≈ particle with inertia rolling downhill
- Adam ≈ adaptive step sizes per parameter (like different masses)
- Learning rate schedules ≈ simulated annealing (cooling schedule)

Key concepts:
- Loss function: What we're minimizing (CrossEntropy for classification)
- Optimizer: How we update parameters
- Scheduler: How learning rate evolves over training
- Early stopping: Avoid overfitting by monitoring validation loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from dataclasses import dataclass
from pathlib import Path

from config import TrainConfig, ExperimentConfig
from logger import Logger


@dataclass
class TrainState:
    """Tracks training progress for checkpointing and early stopping."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_val_acc: float = 0.0
    epochs_without_improvement: int = 0


class Trainer:
    """Handles the training loop with logging, checkpointing, and early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ExperimentConfig,
        logger: Logger,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.device = device
        
        # Loss function (CrossEntropy for classification)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.state = TrainState()
        
        # Checkpoint directory
        if cfg.log.save_checkpoints:
            self.checkpoint_dir = Path(cfg.log.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> dict:
        """Run full training loop.
        
        Returns:
            Dictionary with final metrics
        """
        print(f"\nStarting training for {self.cfg.train.num_epochs} epochs")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Device: {self.device}")
        print()
        
        for epoch in range(self.cfg.train.num_epochs):
            self.state.epoch = epoch
            
            # Training epoch
            train_loss, train_acc = self._train_epoch()
            
            # Validation
            val_loss, val_acc = self._validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch metrics
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                train_acc=train_acc,
                val_acc=val_acc,
            )
            
            # Print progress
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            # Early stopping check (based on validation accuracy)
            if val_acc > self.state.best_val_acc + self.cfg.train.min_delta:
                self.state.best_val_acc = val_acc
                self.state.best_val_loss = val_loss
                self.state.epochs_without_improvement = 0
                
                # Save best model
                if self.cfg.log.save_checkpoints:
                    self._save_checkpoint("best.pt")
            else:
                self.state.epochs_without_improvement += 1
                
                if self.state.epochs_without_improvement >= self.cfg.train.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.cfg.train.patience} epochs)")
                    break
        
        # Save final checkpoint
        if self.cfg.log.save_checkpoints:
            self._save_checkpoint("final.pt")
        
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
            "best_val_acc": self.state.best_val_acc,
            "epochs_trained": self.state.epoch + 1,
        }
    
    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch.
        
        Returns:
            (average loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Optional: Gradient clipping (helps stability)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
            
            self.state.global_step += 1
            
            # Log batch metrics periodically
            if self.state.global_step % self.cfg.log.log_every_n_steps == 0:
                self.logger.log_step(
                    step=self.state.global_step,
                    loss=loss.item(),
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                )
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Run validation.
        
        Returns:
            (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
        
        return total_loss / total, correct / total
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.state.best_val_loss,
            'best_val_acc': self.state.best_val_acc,
            'config': self.cfg,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.state.epoch = checkpoint['epoch']
        self.state.global_step = checkpoint['global_step']
        self.state.best_val_loss = checkpoint['best_val_loss']
        self.state.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.state.epoch}")


# ============================================================================
# Optimizer and scheduler factories
# ============================================================================

def create_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    """Create optimizer from config."""
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def create_scheduler(
    optimizer: Optimizer, 
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> LRScheduler | None:
    """Create learning rate scheduler from config.
    
    Physics analogy:
    - Cosine annealing: Smooth cooling schedule, good for finding minimum
    - Step decay: Sudden temperature drops, can escape local minima
    """
    if cfg.scheduler == "none":
        return None
    
    total_steps = cfg.num_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    
    if cfg.scheduler == "cosine":
        # Cosine annealing with linear warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif cfg.scheduler == "step":
        # Step decay every 1/3 of training
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.num_epochs // 3,
            gamma=0.1,
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# ============================================================================
# Evaluation utilities
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Comprehensive evaluation on a dataset.
    
    Returns:
        Dictionary with various metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
        
        all_predictions.append(predicted.cpu())
        all_targets.append(targets.cpu())
    
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    # Per-class accuracy
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    for pred, target in zip(predictions, targets):
        class_total[target] += 1
        if pred == target:
            class_correct[target] += 1
    
    per_class_acc = {i: (class_correct[i] / class_total[i]).item() 
                     for i in range(10) if class_total[i] > 0}
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'per_class_accuracy': per_class_acc,
        'predictions': predictions,
        'targets': targets,
    }