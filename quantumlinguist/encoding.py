from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalizes a vector, adding small perturbation if near-zero."""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        vec = vec + 1e-8 * np.random.randn(*vec.shape)
        norm = np.linalg.norm(vec)
    return vec / norm


class BERTEncoder:
    """Encodes words using BERT contextual embeddings with optional projection."""

    def __init__(self, model_name: str = "bert-base-uncased", compress_to: int = 4, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModel
        assert compress_to > 0 and (compress_to & (compress_to - 1)) == 0, "compress_to must be a power of 2"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.projection = nn.Linear(768, compress_to).to(device)
        logger.debug("BERTEncoder loaded model: %s", model_name)

    def encode(self, words: List[str], context: str) -> np.ndarray:
        """
        Encodes words using BERT hidden states from the given context.

        Returns shape (n_words, compress_to) as a numpy array.
        """
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True).to(self.device)
        tokens = self.tokenizer.tokenize(context)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)

        results = []
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            positions = []
            for i in range(len(tokens) - len(word_tokens) + 1):
                if tokens[i:i+len(word_tokens)] == word_tokens:
                    positions = list(range(i + 1, i + 1 + len(word_tokens)))
                    break
            if positions:
                vec = hidden[positions].mean(dim=0)
            else:
                logger.warning("Word '%s' not found in context tokens; using mean.", word)
                vec = hidden.mean(dim=0)
            projected = self.projection(vec.unsqueeze(0)).squeeze(0)
            results.append(projected.cpu().numpy())

        return np.stack(results, axis=0)

    def amplitude_encode(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalizes a vector for amplitude encoding, perturbing zero vectors."""
        return l2_normalize(vector)

    def angle_encode(self, vector: np.ndarray) -> np.ndarray:
        """Applies arctan element-wise to produce angle encodings."""
        return np.arctan(vector)


class GloVeEncoder:
    """Encodes words using pre-trained GloVe vectors."""

    def __init__(self, glove_path: str, dim: int = 50):
        self.dim = dim
        self._vectors: dict[str, np.ndarray] = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                try:
                    self._vectors[word] = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    logger.warning("Skipping malformed GloVe line for token: %s", word)
        logger.debug("GloVeEncoder loaded %d vectors from %s", len(self._vectors), glove_path)

    def encode(self, words: List[str]) -> np.ndarray:
        """
        Encodes a list of words using GloVe vectors.

        Returns L2-normalized shape (n_words, dim).
        """
        result = []
        for word in words:
            vec = self._vectors.get(word.lower())
            if vec is None:
                logger.debug("OOV word in GloVe: %s", word)
                vec = self._oov_vector(word)
            result.append(l2_normalize(vec))
        return np.stack(result, axis=0)

    def _oov_vector(self, word: str) -> np.ndarray:
        """Returns a character trigram average vector, falling back to randn."""
        trigrams = [word[i:i+3] for i in range(max(1, len(word) - 2))]
        vecs = [self._vectors[t] for t in trigrams if t in self._vectors]
        if vecs:
            return np.mean(vecs, axis=0)
        logger.warning("No trigram fallback found for '%s'; using random vector.", word)
        return np.random.randn(self.dim).astype(np.float32)


class LearnedEncoder(nn.Module):
    """Learned word embedding encoder backed by nn.Embedding."""

    def __init__(self, vocab_size: int, embed_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.word2idx: dict[str, int] = {}

    def encode(self, words: List[str]) -> np.ndarray:
        """
        Returns L2-normalized embeddings of shape (n_words, embed_dim).

        Builds vocabulary lazily on first call.
        """
        if not self.word2idx:
            self.word2idx = {w: i for i, w in enumerate(set(words))}
            logger.debug("LearnedEncoder built vocab of size %d", len(self.word2idx))

        indices = []
        for word in words:
            idx = self.word2idx.get(word)
            if idx is None:
                logger.warning("Word '%s' not in learned vocab; using index 0.", word)
                idx = 0
            indices.append(idx)

        tensor = torch.tensor(indices, dtype=torch.long)
        with torch.no_grad():
            vecs = self.forward(tensor).numpy()

        return np.stack([l2_normalize(v) for v in vecs], axis=0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)
