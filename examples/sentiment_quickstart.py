"""Minimal sentiment classification quickstart."""
from quantumlinguist import QuantumSentimentClassifier, EntanglementAnalyzer
from quantumlinguist.benchmark import BenchmarkSuite

train_x, train_y, test_x, test_y = BenchmarkSuite.sentiment_sst2(n_samples=100)

model = QuantumSentimentClassifier(backend="default.qubit")
model.fit(train_x[:80], train_y[:80], epochs=5)

metrics = model.evaluate(test_x, test_y)
print(f"Accuracy: {metrics['accuracy']:.3f}")

analyzer = EntanglementAnalyzer(model)
report = analyzer.explain("The film was surprisingly emotional")
print(report)

from quantumlinguist.analysis import print_entanglement_map
print_entanglement_map(report["entanglement_map"])
