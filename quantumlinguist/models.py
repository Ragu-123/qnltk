from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json, os, logging
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from quantumlinguist.parser import NLTKParser, ParseNode
from quantumlinguist.compiler import compile as ql_compile, CircuitMetadata
from quantumlinguist.encoding import LearnedEncoder

logger = logging.getLogger(__name__)


class QuantumNLPBase:
    def fit(self, texts, labels, epochs=20, lr=0.01, batch_size=8, verbose=True):
        """Train the model on texts and labels."""
        raise NotImplementedError

    def predict(self, texts) -> List[Dict]:
        """Return predictions for a list of texts."""
        raise NotImplementedError

    def evaluate(self, texts, labels) -> Dict:
        """Evaluate the model and return metrics."""
        raise NotImplementedError

    def save(self, path: str):
        """Save model weights and config to path."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        """Load and return a model instance from path."""
        raise NotImplementedError

    def get_circuit(self, text: str) -> Tuple:
        """Return (qnode, CircuitMetadata) for the given text."""
        raise NotImplementedError


class QuantumSentimentClassifier(QuantumNLPBase):
    def __init__(self, backend="default.qubit", ruleset="constituency_v1", embed_dim=8, vocab_size=1000, lr=0.01):
        self.backend = backend
        self.ruleset = ruleset
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.lr = lr
        self.parser = NLTKParser()
        self.encoder = LearnedEncoder(vocab_size, embed_dim)
        self.word2idx: Dict[str, int] = {}
        self.params: Optional[torch.Tensor] = None
        self._max_params: int = 0

    def _build_vocab(self, texts):
        """Collect unique words from texts and assign indices to the encoder."""
        words = []
        for text in texts:
            words.extend(text.lower().split())
        unique = sorted(set(words))
        self.word2idx = {w: i % self.vocab_size for i, w in enumerate(unique)}

    def _make_qnode(self, n_qubits: int):
        dev = qml.device(self.backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i] if i < len(params) else torch.tensor(0.0), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def _text_to_circuit_output(self, text: str) -> Tuple[torch.Tensor, CircuitMetadata]:
        """Parse text, encode words, build QNode, return (expvals_tensor, CircuitMetadata)."""
        node = self.parser.parse(text)
        words = text.lower().split()
        if not words:
            words = ["unknown"]
        idxs = torch.tensor([self.word2idx.get(w, 0) for w in words], dtype=torch.long)
        embeddings = self.encoder.forward(idxs)
        angles = embeddings.mean(dim=-1)

        qnode, meta = ql_compile(node, self.ruleset, self.backend)
        n_qubits = meta.n_qubits if meta.n_qubits > 0 else len(words)
        n_params = max(n_qubits, len(angles))

        if self.params is None or self.params.shape[0] < n_params:
            init = torch.zeros(n_params)
            init[: len(angles)] = angles.detach()
            self.params = nn.Parameter(init)
            self._max_params = n_params

        params_slice = self.params[:n_qubits]
        circuit = self._make_qnode(n_qubits)
        result = circuit(params_slice)
        expvals = torch.stack(result) if isinstance(result, (list, tuple)) else result
        return expvals, meta

    def fit(self, texts, labels, epochs=20, lr=0.01, batch_size=8, verbose=True):
        """Train on texts (list of str) with binary labels (0/1) using Adam + BCEWithLogitsLoss."""
        self._build_vocab(texts)
        sample_node = self.parser.parse(texts[0])
        _, meta = ql_compile(sample_node, self.ruleset, self.backend)
        n_qubits = meta.n_qubits if meta.n_qubits > 0 else len(texts[0].split())
        self._max_params = n_qubits
        self.params = nn.Parameter(torch.randn(n_qubits))

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + [self.params], lr=lr
        )
        loss_fn = nn.BCEWithLogitsLoss()
        self._train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = list(range(len(texts)))
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i: i + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True)
                for idx in batch_idx:
                    try:
                        expvals, _ = self._text_to_circuit_output(texts[idx])
                        logit = expvals.mean().unsqueeze(0)
                        target = torch.tensor([float(labels[idx])])
                        loss = loss_fn(logit, target)
                        batch_loss = batch_loss + loss
                    except Exception as e:
                        logger.warning(f"Skipping sample {idx}: {e}")
                batch_loss = batch_loss / max(len(batch_idx), 1)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            avg = epoch_loss / max(1, len(texts) // batch_size)
            self._train_losses.append(avg)
            if verbose:
                logger.info(f"Epoch {epoch + 1}/{epochs} loss={avg:.4f}")

    def predict(self, texts) -> List[Dict]:
        """Return list of dicts with label, confidence, circuit_depth, n_qubits."""
        results = []
        with torch.no_grad():
            for text in texts:
                try:
                    expvals, meta = self._text_to_circuit_output(text)
                    logit = expvals.mean().item()
                    prob = torch.sigmoid(torch.tensor(logit)).item()
                    results.append({
                        "label": "positive" if prob >= 0.5 else "negative",
                        "confidence": prob if prob >= 0.5 else 1.0 - prob,
                        "circuit_depth": meta.depth,
                        "n_qubits": meta.n_qubits,
                    })
                except Exception as e:
                    logger.warning(f"predict failed for text: {e}")
                    results.append({"label": "negative", "confidence": 0.5, "circuit_depth": 0, "n_qubits": 0})
        return results

    def evaluate(self, texts, labels) -> Dict:
        """Return accuracy and F1 score."""
        preds = self.predict(texts)
        pred_labels = [1 if p["label"] == "positive" else 0 for p in preds]
        correct = sum(p == l for p, l in zip(pred_labels, labels))
        acc = correct / len(labels) if labels else 0.0
        tp = sum(p == 1 and l == 1 for p, l in zip(pred_labels, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(pred_labels, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(pred_labels, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"accuracy": acc, "f1": f1}

    def save(self, path: str):
        """Save encoder state_dict and config as JSON sidecar."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        if self.params is not None:
            torch.save(self.params.data, os.path.join(path, "params.pt"))
        config = {
            "backend": self.backend,
            "ruleset": self.ruleset,
            "embed_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
            "lr": self.lr,
            "word2idx": self.word2idx,
            "max_params": self._max_params,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str):
        """Load and return a QuantumSentimentClassifier from path."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = cls(
            backend=config["backend"],
            ruleset=config["ruleset"],
            embed_dim=config["embed_dim"],
            vocab_size=config["vocab_size"],
            lr=config["lr"],
        )
        obj.word2idx = config["word2idx"]
        obj._max_params = config["max_params"]
        obj.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt")))
        params_path = os.path.join(path, "params.pt")
        if os.path.exists(params_path):
            obj.params = nn.Parameter(torch.load(params_path))
        return obj

    def get_circuit(self, text: str) -> Tuple:
        """Return (qnode, CircuitMetadata) for the given text."""
        node = self.parser.parse(text)
        qnode, meta = ql_compile(node, self.ruleset, self.backend)
        return qnode, meta


class QuantumTextualEntailment(QuantumNLPBase):
    def __init__(self, backend="default.qubit", ruleset="constituency_v1", embed_dim=8, vocab_size=1000):
        self.backend = backend
        self.ruleset = ruleset
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.parser = NLTKParser()
        self.encoder = LearnedEncoder(vocab_size, embed_dim)
        self.word2idx: Dict[str, int] = {}
        self.params_p: Optional[nn.Parameter] = None
        self.params_h: Optional[nn.Parameter] = None
        self.head: Optional[nn.Linear] = None

    def _build_vocab(self, texts):
        """Collect unique words from premise/hypothesis pairs."""
        words = []
        for prem, hyp in texts:
            words.extend(prem.lower().split())
            words.extend(hyp.lower().split())
        unique = sorted(set(words))
        self.word2idx = {w: i % self.vocab_size for i, w in enumerate(unique)}

    def _make_qnode(self, n_qubits: int):
        dev = qml.device(self.backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i] if i < len(params) else torch.tensor(0.0), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def _encode_sentence(self, text: str, params: nn.Parameter) -> Tuple[torch.Tensor, CircuitMetadata]:
        node = self.parser.parse(text)
        words = text.lower().split() or ["unknown"]
        idxs = torch.tensor([self.word2idx.get(w, 0) for w in words], dtype=torch.long)
        embeddings = self.encoder.forward(idxs)
        qnode, meta = ql_compile(node, self.ruleset, self.backend)
        n_qubits = meta.n_qubits if meta.n_qubits > 0 else len(words)
        p = params[:n_qubits]
        circuit = self._make_qnode(n_qubits)
        result = circuit(p)
        expvals = torch.stack(result) if isinstance(result, (list, tuple)) else result
        return expvals, meta

    def fit(self, texts, labels, epochs=20, lr=0.01, batch_size=8, verbose=True):
        """Train on (premise, hypothesis) pairs with 3-class labels (0=entailment,1=neutral,2=contradiction)."""
        self._build_vocab(texts)
        n_qubits = 4
        self.params_p = nn.Parameter(torch.randn(n_qubits))
        self.params_h = nn.Parameter(torch.randn(n_qubits))
        self.head = nn.Linear(n_qubits * 2, 3)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + [self.params_p, self.params_h] + list(self.head.parameters()), lr=lr
        )
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                batch_labels = labels[i: i + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True)
                for (prem, hyp), lbl in zip(batch, batch_labels):
                    try:
                        ev_p, _ = self._encode_sentence(prem, self.params_p)
                        ev_h, _ = self._encode_sentence(hyp, self.params_h)
                        n = min(len(ev_p), len(ev_h), n_qubits)
                        combined = torch.cat([ev_p[:n], ev_h[:n]])
                        pad = n_qubits * 2 - len(combined)
                        if pad > 0:
                            combined = torch.cat([combined, torch.zeros(pad)])
                        logits = self.head(combined.unsqueeze(0))
                        target = torch.tensor([lbl])
                        loss = loss_fn(logits, target)
                        batch_loss = batch_loss + loss
                    except Exception as e:
                        logger.warning(f"Skipping NLI sample: {e}")
                batch_loss = batch_loss / max(len(batch), 1)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            if verbose:
                logger.info(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")

    def predict(self, texts) -> List[Dict]:
        """Return list of dicts with label, confidence, circuit_depth."""
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        results = []
        n_qubits = 4
        with torch.no_grad():
            for prem, hyp in texts:
                try:
                    ev_p, meta_p = self._encode_sentence(prem, self.params_p)
                    ev_h, meta_h = self._encode_sentence(hyp, self.params_h)
                    n = min(len(ev_p), len(ev_h), n_qubits)
                    combined = torch.cat([ev_p[:n], ev_h[:n]])
                    pad = n_qubits * 2 - len(combined)
                    if pad > 0:
                        combined = torch.cat([combined, torch.zeros(pad)])
                    logits = self.head(combined.unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1).squeeze()
                    pred = int(probs.argmax().item())
                    results.append({
                        "label": label_map[pred],
                        "confidence": probs[pred].item(),
                        "circuit_depth": max(meta_p.depth, meta_h.depth),
                    })
                except Exception as e:
                    logger.warning(f"predict NLI failed: {e}")
                    results.append({"label": "neutral", "confidence": 0.33, "circuit_depth": 0})
        return results

    def evaluate(self, texts, labels) -> Dict:
        """Return accuracy and F1 (macro) over 3 classes."""
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        preds = [label_map[p["label"]] for p in self.predict(texts)]
        correct = sum(p == l for p, l in zip(preds, labels))
        acc = correct / len(labels) if labels else 0.0
        f1_sum = 0.0
        for cls in range(3):
            tp = sum(p == cls and l == cls for p, l in zip(preds, labels))
            fp = sum(p == cls and l != cls for p, l in zip(preds, labels))
            fn = sum(p != cls and l == cls for p, l in zip(preds, labels))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_sum += 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"accuracy": acc, "f1": f1_sum / 3}

    def save(self, path: str):
        """Save encoder, head, params and config."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        if self.head is not None:
            torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))
        if self.params_p is not None:
            torch.save(self.params_p.data, os.path.join(path, "params_p.pt"))
        if self.params_h is not None:
            torch.save(self.params_h.data, os.path.join(path, "params_h.pt"))
        config = {
            "backend": self.backend, "ruleset": self.ruleset,
            "embed_dim": self.embed_dim, "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str):
        """Load and return a QuantumTextualEntailment instance from path."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = cls(backend=config["backend"], ruleset=config["ruleset"],
                  embed_dim=config["embed_dim"], vocab_size=config["vocab_size"])
        obj.word2idx = config["word2idx"]
        obj.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt")))
        obj.head = nn.Linear(8, 3)
        obj.head.load_state_dict(torch.load(os.path.join(path, "head.pt")))
        obj.params_p = nn.Parameter(torch.load(os.path.join(path, "params_p.pt")))
        obj.params_h = nn.Parameter(torch.load(os.path.join(path, "params_h.pt")))
        return obj

    def get_circuit(self, text: str) -> Tuple:
        """Return (qnode, CircuitMetadata) for the given text."""
        node = self.parser.parse(text)
        return ql_compile(node, self.ruleset, self.backend)


class QuantumSemanticSimilarity(QuantumNLPBase):
    def __init__(self, backend="default.qubit", ruleset="constituency_v1", embed_dim=8, vocab_size=1000):
        self.backend = backend
        self.ruleset = ruleset
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.parser = NLTKParser()
        self.encoder = LearnedEncoder(vocab_size, embed_dim)
        self.word2idx: Dict[str, int] = {}
        self.params_a: Optional[nn.Parameter] = None
        self.params_b: Optional[nn.Parameter] = None

    def _build_vocab(self, texts):
        """Collect unique words from sentence pairs."""
        words = []
        for s1, s2 in texts:
            words.extend(s1.lower().split())
            words.extend(s2.lower().split())
        unique = sorted(set(words))
        self.word2idx = {w: i % self.vocab_size for i, w in enumerate(unique)}

    def _make_qnode(self, n_qubits: int):
        dev = qml.device(self.backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i] if i < len(params) else torch.tensor(0.0), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def _sentence_vector(self, text: str, params: nn.Parameter) -> Tuple[torch.Tensor, CircuitMetadata]:
        node = self.parser.parse(text)
        words = text.lower().split() or ["unknown"]
        idxs = torch.tensor([self.word2idx.get(w, 0) for w in words], dtype=torch.long)
        self.encoder.forward(idxs)
        qnode, meta = ql_compile(node, self.ruleset, self.backend)
        n_qubits = meta.n_qubits if meta.n_qubits > 0 else len(words)
        p = params[:n_qubits]
        circuit = self._make_qnode(n_qubits)
        result = circuit(p)
        expvals = torch.stack(result) if isinstance(result, (list, tuple)) else result
        return expvals, meta

    def fit(self, texts, labels, epochs=20, lr=0.01, batch_size=8, verbose=True):
        """Train on (sent1, sent2) pairs with continuous similarity labels in [0,1] using MSELoss."""
        self._build_vocab(texts)
        n_qubits = 4
        self.params_a = nn.Parameter(torch.randn(n_qubits))
        self.params_b = nn.Parameter(torch.randn(n_qubits))
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + [self.params_a, self.params_b], lr=lr
        )
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                batch_labels = labels[i: i + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True)
                for (s1, s2), lbl in zip(batch, batch_labels):
                    try:
                        ev_a, _ = self._sentence_vector(s1, self.params_a)
                        ev_b, _ = self._sentence_vector(s2, self.params_b)
                        n = min(len(ev_a), len(ev_b))
                        sim = torch.dot(ev_a[:n], ev_b[:n]) / (n + 1e-8)
                        sim_scaled = (sim + 1) / 2
                        target = torch.tensor(float(lbl))
                        loss = loss_fn(sim_scaled, target)
                        batch_loss = batch_loss + loss
                    except Exception as e:
                        logger.warning(f"Skipping similarity sample: {e}")
                batch_loss = batch_loss / max(len(batch), 1)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            if verbose:
                logger.info(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")

    def predict(self, texts) -> List[Dict]:
        """Return list of dicts with similarity, circuit_depth, n_qubits."""
        results = []
        with torch.no_grad():
            for s1, s2 in texts:
                try:
                    ev_a, meta_a = self._sentence_vector(s1, self.params_a)
                    ev_b, meta_b = self._sentence_vector(s2, self.params_b)
                    n = min(len(ev_a), len(ev_b))
                    sim = torch.dot(ev_a[:n], ev_b[:n]) / (n + 1e-8)
                    sim_val = ((sim + 1) / 2).item()
                    results.append({
                        "similarity": float(np.clip(sim_val, 0.0, 1.0)),
                        "circuit_depth": max(meta_a.depth, meta_b.depth),
                        "n_qubits": max(meta_a.n_qubits, meta_b.n_qubits),
                    })
                except Exception as e:
                    logger.warning(f"predict similarity failed: {e}")
                    results.append({"similarity": 0.5, "circuit_depth": 0, "n_qubits": 0})
        return results

    def evaluate(self, texts, labels) -> Dict:
        """Return MSE and Pearson correlation as accuracy proxy."""
        preds = [p["similarity"] for p in self.predict(texts)]
        mse = float(np.mean([(p - l) ** 2 for p, l in zip(preds, labels)]))
        if len(preds) > 1:
            corr = float(np.corrcoef(preds, labels)[0, 1])
        else:
            corr = 0.0
        return {"accuracy": corr, "f1": 1.0 - mse}

    def save(self, path: str):
        """Save encoder, params and config."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        if self.params_a is not None:
            torch.save(self.params_a.data, os.path.join(path, "params_a.pt"))
        if self.params_b is not None:
            torch.save(self.params_b.data, os.path.join(path, "params_b.pt"))
        config = {
            "backend": self.backend, "ruleset": self.ruleset,
            "embed_dim": self.embed_dim, "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str):
        """Load and return a QuantumSemanticSimilarity instance from path."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = cls(backend=config["backend"], ruleset=config["ruleset"],
                  embed_dim=config["embed_dim"], vocab_size=config["vocab_size"])
        obj.word2idx = config["word2idx"]
        obj.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt")))
        obj.params_a = nn.Parameter(torch.load(os.path.join(path, "params_a.pt")))
        obj.params_b = nn.Parameter(torch.load(os.path.join(path, "params_b.pt")))
        return obj

    def get_circuit(self, text: str) -> Tuple:
        """Return (qnode, CircuitMetadata) for the given text."""
        node = self.parser.parse(text)
        return ql_compile(node, self.ruleset, self.backend)


class QuantumNER(QuantumNLPBase):
    NER_LABELS = ["PER", "ORG", "LOC", "O"]

    def __init__(self, backend="default.qubit", ruleset="constituency_v1", embed_dim=8, vocab_size=1000):
        self.backend = backend
        self.ruleset = ruleset
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.parser = NLTKParser()
        self.encoder = LearnedEncoder(vocab_size, embed_dim)
        self.word2idx: Dict[str, int] = {}
        self.params: Optional[nn.Parameter] = None
        self.token_head: Optional[nn.Linear] = None

    def _build_vocab(self, texts):
        """Collect unique words from all texts."""
        words = []
        for text in texts:
            words.extend(text.lower().split())
        unique = sorted(set(words))
        self.word2idx = {w: i % self.vocab_size for i, w in enumerate(unique)}

    def _make_qnode(self, n_qubits: int):
        dev = qml.device(self.backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i] if i < len(params) else torch.tensor(0.0), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def fit(self, texts, labels, epochs=20, lr=0.01, batch_size=8, verbose=True):
        """Train on texts with per-token NER label sequences (lists of ints 0-3)."""
        self._build_vocab(texts)
        n_qubits = 4
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.token_head = nn.Linear(1, 4)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + [self.params] + list(self.token_head.parameters()), lr=lr
        )
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_labels = labels[i: i + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True)
                for text, token_labels in zip(batch_texts, batch_labels):
                    try:
                        node = self.parser.parse(text)
                        words = text.lower().split() or ["unknown"]
                        idxs = torch.tensor([self.word2idx.get(w, 0) for w in words], dtype=torch.long)
                        self.encoder.forward(idxs)
                        qnode, meta = ql_compile(node, self.ruleset, self.backend)
                        n_q = meta.n_qubits if meta.n_qubits > 0 else len(words)
                        p = self.params[:n_q]
                        circuit = self._make_qnode(n_q)
                        result = circuit(p)
                        expvals = torch.stack(result) if isinstance(result, (list, tuple)) else result
                        for t_idx, t_lbl in enumerate(token_labels):
                            ev = expvals[t_idx % len(expvals)].unsqueeze(0).unsqueeze(0)
                            logits = self.token_head(ev)
                            target = torch.tensor([t_lbl])
                            loss = loss_fn(logits, target)
                            batch_loss = batch_loss + loss
                    except Exception as e:
                        logger.warning(f"Skipping NER sample: {e}")
                batch_loss = batch_loss / max(len(batch_texts), 1)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            if verbose:
                logger.info(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")

    def predict(self, texts) -> List[List[Dict]]:
        """Return list of per-sentence token-label dicts with word, label, confidence."""
        results = []
        with torch.no_grad():
            for text in texts:
                try:
                    node = self.parser.parse(text)
                    words = text.lower().split() or ["unknown"]
                    idxs = torch.tensor([self.word2idx.get(w, 0) for w in words], dtype=torch.long)
                    self.encoder.forward(idxs)
                    qnode, meta = ql_compile(node, self.ruleset, self.backend)
                    n_q = meta.n_qubits if meta.n_qubits > 0 else len(words)
                    p = self.params[:n_q]
                    circuit = self._make_qnode(n_q)
                    result = circuit(p)
                    expvals = torch.stack(result) if isinstance(result, (list, tuple)) else result
                    token_preds = []
                    for t_idx, word in enumerate(words):
                        ev = expvals[t_idx % len(expvals)].unsqueeze(0).unsqueeze(0)
                        logits = self.token_head(ev)
                        probs = torch.softmax(logits, dim=-1).squeeze()
                        pred = int(probs.argmax().item())
                        token_preds.append({
                            "word": word,
                            "label": self.NER_LABELS[pred],
                            "confidence": probs[pred].item(),
                        })
                    results.append(token_preds)
                except Exception as e:
                    logger.warning(f"predict NER failed: {e}")
                    results.append([{"word": w, "label": "O", "confidence": 0.25} for w in text.split()])
        return results

    def evaluate(self, texts, labels) -> Dict:
        """Return token-level accuracy and F1."""
        label_map = {l: i for i, l in enumerate(self.NER_LABELS)}
        preds_all, labels_all = [], []
        for sent_preds, sent_labels in zip(self.predict(texts), labels):
            for p, l in zip(sent_preds, sent_labels):
                preds_all.append(label_map.get(p["label"], 3))
                labels_all.append(l)
        correct = sum(p == l for p, l in zip(preds_all, labels_all))
        acc = correct / len(labels_all) if labels_all else 0.0
        f1_sum = 0.0
        for cls in range(4):
            tp = sum(p == cls and l == cls for p, l in zip(preds_all, labels_all))
            fp = sum(p == cls and l != cls for p, l in zip(preds_all, labels_all))
            fn = sum(p != cls and l == cls for p, l in zip(preds_all, labels_all))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_sum += 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"accuracy": acc, "f1": f1_sum / 4}

    def save(self, path: str):
        """Save encoder, token_head, params and config."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        if self.token_head is not None:
            torch.save(self.token_head.state_dict(), os.path.join(path, "token_head.pt"))
        if self.params is not None:
            torch.save(self.params.data, os.path.join(path, "params.pt"))
        config = {
            "backend": self.backend, "ruleset": self.ruleset,
            "embed_dim": self.embed_dim, "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str):
        """Load and return a QuantumNER instance from path."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = cls(backend=config["backend"], ruleset=config["ruleset"],
                  embed_dim=config["embed_dim"], vocab_size=config["vocab_size"])
        obj.word2idx = config["word2idx"]
        obj.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt")))
        obj.token_head = nn.Linear(1, 4)
        obj.token_head.load_state_dict(torch.load(os.path.join(path, "token_head.pt")))
        obj.params = nn.Parameter(torch.load(os.path.join(path, "params.pt")))
        return obj

    def get_circuit(self, text: str) -> Tuple:
        """Return (qnode, CircuitMetadata) for the given text."""
        node = self.parser.parse(text)
        return ql_compile(node, self.ruleset, self.backend)
