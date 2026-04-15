from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json, time, logging
import numpy as np
import torch

from quantumlinguist.models import QuantumNLPBase

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    avg_circuit_depth: List[float] = field(default_factory=list)
    avg_entanglement_entropy: List[float] = field(default_factory=list)


class SPSAOptimizer:
    def __init__(self, lr=0.01, perturbation=0.1):
        self.lr = lr
        self.c = perturbation

    def step(self, loss_fn: callable, params: torch.Tensor) -> torch.Tensor:
        """One SPSA step: perturb all params +-delta, estimate gradient, update."""
        delta = torch.bernoulli(torch.ones_like(params) * 0.5) * 2 - 1
        loss_plus = loss_fn(params + self.c * delta)
        loss_minus = loss_fn(params - self.c * delta)
        grad_approx = (loss_plus - loss_minus) / (2 * self.c * delta)
        return params - self.lr * grad_approx


class Trainer:
    def __init__(
        self,
        model: QuantumNLPBase,
        optimizer: str = "adam",
        lr: float = 0.01,
        seed: int = 42,
        log_entanglement: bool = False,
    ):
        self.model = model
        self.optimizer_name = optimizer
        self.lr = lr
        self.seed = seed
        self.log_entanglement = log_entanglement
        self.history = TrainingHistory()

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, epochs=20) -> TrainingHistory:
        """Run training loop. Returns TrainingHistory."""
        torch.manual_seed(self.seed)
        self.model.fit(train_texts, train_labels, epochs=epochs, lr=self.lr, verbose=True)
        if hasattr(self.model, "_train_losses"):
            self.history.train_loss = self.model._train_losses
        if val_texts is not None and val_labels is not None:
            val_metrics = self.model.evaluate(val_texts, val_labels)
            self.history.val_acc = [val_metrics.get("accuracy", 0.0)]
        return self.history

    def save_artifact(self, path: str) -> None:
        """Save full reproducibility JSON artifact."""
        artifact = {
            "seed": self.seed,
            "optimizer": self.optimizer_name,
            "lr": self.lr,
            "model_class": self.model.__class__.__name__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "history": {
                "train_loss": self.history.train_loss,
                "val_loss": self.history.val_loss,
                "train_acc": self.history.train_acc,
                "val_acc": self.history.val_acc,
            },
        }
        with open(path, "w") as f:
            json.dump(artifact, f, indent=2)

    @classmethod
    def reproduce(cls, artifact_path: str) -> dict:
        """Load artifact and recreate training config."""
        with open(artifact_path) as f:
            artifact = json.load(f)
        logger.info(f"Loaded artifact: {artifact['model_class']} seed={artifact['seed']}")
        return artifact
