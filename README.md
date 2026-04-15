# QuantumLinguist 

**QuantumLinguist** is a grammar-driven hybrid quantum-classical Natural Language Processing (QNLP) research library. It provides a modular framework for mapping formal linguistic structures directly into quantum circuits using **PennyLane** and **NLTK**.

By treating the syntax of a sentence as the template for a quantum circuit, QuantumLinguist allows researchers to explore the intersection of category theory, linguistics, and quantum machine learning.

---

## Features

- **Grammar-to-Circuit Compilation**: Automatically transform NLTK parse trees into parameterized quantum circuits (PQCs) using customizable rulesets.
- **Hybrid Neural Architectures**: Seamlessly combine classical encoders (BERT, GloVe) with quantum layers.
- **Task-Specific QNLP Models**: Out-of-the-box support for:
  - Sentiment Classification
  - Textual Entailment
  - Semantic Similarity
  - Named Entity Recognition (NER)
- **Quantum Interpretability**: Tools to analyze entanglement entropy and circuit depth to "explain" how the quantum model processes linguistic relationships.
- **Extensible Encoding**: Map words to quantum states via learned parameters or pre-trained transformer embeddings.

---

## Architecture

The QuantumLinguist pipeline follows a four-stage process:

1.  **Parsing**: Text is converted into a constituency or dependency tree using `NLTKParser`.
2.  **Encoding**: Words are mapped to numerical vectors using `BERTEncoder` or `LearnedEncoder`.
3.  **Compilation**: The `compiler` traverses the tree and generates a PennyLane circuit where linguistic dependencies (like Subject-Verb-Object) are mapped to quantum gates (like CNOT or Controlled-Rotations).
4.  **Execution**: The circuit is executed on quantum simulators (or hardware) to produce predictions.

---

## Installation

### Prerequisites
- Python 3.10+
- (Optional) CUDA-enabled GPU for `pennylane-lightning[gpu]`

### Setup
Clone the repository and install the package in editable mode:

```powershell
https://github.com/Ragu-123/qnltk.git
cd quantumlinguist
pip install -e ".[dev]"
```

The installer will automatically download necessary NLTK data (punkt, averaged_perceptron_tagger, etc.) upon first run of the parser.

---

## Quickstart: Sentiment Analysis

Train a quantum sentiment classifier in just a few lines:

```python
from quantumlinguist import QuantumSentimentClassifier, EntanglementAnalyzer
from quantumlinguist.benchmark import BenchmarkSuite

# 1. Load a small research dataset (SST-2)
train_x, train_y, test_x, test_y = BenchmarkSuite.sentiment_sst2(n_samples=100)

# 2. Initialize the Quantum Model
model = QuantumSentimentClassifier(backend="default.qubit")

# 3. Train using hybrid quantum-classical optimization
model.fit(train_x[:80], train_y[:80], epochs=5)

# 4. Explain a prediction using Entanglement Analysis
analyzer = EntanglementAnalyzer(model)
explanation = analyzer.explain("The film was surprisingly emotional")
print(f"Entanglement Map: {explanation['entanglement_map']}")
```

---

## Running Examples

The `examples/` directory contains several scripts to get you started:

- **Basic Sentiment**: 
  ```powershell
  python examples/sentiment_quickstart.py
  ```
- **Interpretability Demo**: Visualize how the quantum state changes across a sentence.
  ```powershell
  python examples/interpretability_demo.py
  ```
- **Entailment**: Determine if one sentence logically follows another using quantum interference.
  ```powershell
  python examples/entailment_example.py
  ```

---

## Research & Reproducibility

QuantumLinguist is built for reproducibility. The `Trainer` class generates artifacts that capture the exact seed, circuit depth, and entanglement metrics for every run.

```python
from quantumlinguist.trainer import Trainer

trainer = Trainer(model, optimizer="adam", lr=0.01)
history = trainer.train(train_x, train_y, epochs=20)
trainer.save_artifact("experiment_results.json")
```

---

## Project Structure

- `quantumlinguist/`:
  - `parser.py`: NLTK-based linguistic tree generation.
  - `compiler.py`: Logic for mapping trees to PennyLane circuits.
  - `models.py`: Implementations of `QuantumSentimentClassifier`, etc.
  - `encoding.py`: BERT/GloVe/Learned embedding strategies.
  - `analysis.py`: Entanglement and interpretability metrics.
- `data/`: Compositional datasets and JSON benchmarks.
- `tests/`: Unit tests for circuits and parsers.

---

## License

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.
