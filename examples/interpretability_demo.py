"""Entanglement interpretability for word-order sensitive sentences."""
from quantumlinguist import QuantumSentimentClassifier, EntanglementAnalyzer
from quantumlinguist.analysis import print_entanglement_map

pairs = [
    ("cats chase dogs", "dogs chase cats"),
    ("scientists discovered proteins", "proteins discovered scientists"),
    ("the company hired the engineer", "the engineer hired the company"),
    ("lions hunt zebras", "zebras hunt lions"),
    ("teachers love students", "students love teachers"),
]

model = QuantumSentimentClassifier(backend="default.qubit")
# Minimal training on toy data
toy_texts = [p for pair in pairs for p in pair]
toy_labels = [1, 0] * len(pairs)
model.fit(toy_texts, toy_labels, epochs=3, verbose=False)

analyzer = EntanglementAnalyzer(model)

for s1, s2 in pairs:
    print(f"\n=== '{s1}' vs '{s2}' ===")
    e1 = analyzer.entanglement_entropy(s1)
    e2 = analyzer.entanglement_entropy(s2)
    print(f"  Entanglement [{s1}]: {e1}")
    print(f"  Entanglement [{s2}]: {e2}")
