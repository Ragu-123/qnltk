"""Textual entailment on SNLI subset."""
from quantumlinguist import QuantumTextualEntailment
from quantumlinguist.benchmark import BenchmarkSuite

train_x, train_y, test_x, test_y = BenchmarkSuite.entailment_snli(n_samples=50)

model = QuantumTextualEntailment(backend="default.qubit")
model.fit(train_x[:40], train_y[:40], epochs=5)

preds = model.predict(test_x[:3])
for (premise, hyp), pred in zip(test_x[:3], preds):
    _, meta1 = model.get_circuit(premise)
    _, meta2 = model.get_circuit(hyp)
    print(f"Premise ({meta1.n_qubits}q): {premise[:50]}")
    print(f"Hypothesis ({meta2.n_qubits}q): {hyp[:50]}")
    print(f"Prediction: {pred['label']} ({pred['confidence']:.2f})")
    print()
