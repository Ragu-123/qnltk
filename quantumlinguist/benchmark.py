from __future__ import annotations
from typing import List, Tuple, Optional
import json, os, logging
from pathlib import Path

from quantumlinguist.models import QuantumNLPBase

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"


class BenchmarkSuite:
    """Standard QNLP benchmark tasks."""

    @staticmethod
    def sentiment_sst2(n_samples: int = 500) -> Tuple[List, List, List, List]:
        """Load SST-2 subset via HuggingFace datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset("sst2", split="train")
            texts = [r["sentence"] for r in ds.select(range(min(n_samples, len(ds))))]
            labels = [r["label"] for r in ds.select(range(min(n_samples, len(ds))))]
            split = int(len(texts) * 0.8)
            return texts[:split], labels[:split], texts[split:], labels[split:]
        except Exception as e:
            logger.warning(f"SST-2 load failed: {e}")
            return [], [], [], []

    @staticmethod
    def entailment_snli(n_samples: int = 500) -> Tuple[List, List, List, List]:
        """Load SNLI subset."""
        try:
            from datasets import load_dataset
            ds = load_dataset("snli", split="train").filter(lambda x: x["label"] != -1)
            ds = ds.select(range(min(n_samples, len(ds))))
            pairs = [(r["premise"], r["hypothesis"]) for r in ds]
            labels = [r["label"] for r in ds]
            split = int(len(pairs) * 0.8)
            return pairs[:split], labels[:split], pairs[split:], labels[split:]
        except Exception as e:
            logger.warning(f"SNLI load failed: {e}")
            return [], [], [], []

    @staticmethod
    def similarity_sts(n_samples: int = 200) -> Tuple[List, List, List, List]:
        """Load STS-Benchmark subset."""
        try:
            from datasets import load_dataset
            ds = load_dataset("stsb_multi_mt", name="en", split="train")
            ds = ds.select(range(min(n_samples, len(ds))))
            pairs = [(r["sentence1"], r["sentence2"]) for r in ds]
            labels = [r["similarity_score"] / 5.0 for r in ds]
            split = int(len(pairs) * 0.8)
            return pairs[:split], labels[:split], pairs[split:], labels[split:]
        except Exception as e:
            logger.warning(f"STS load failed: {e}")
            return [], [], [], []

    @staticmethod
    def custom_compositional() -> Tuple[List, List, List, List]:
        """Handcrafted compositionality test dataset."""
        data_file = _DATA_DIR / "compositional_dataset.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {data_file}")
        with open(data_file) as f:
            data = json.load(f)
        texts = [d["sentence"] for d in data]
        labels = [d["label"] for d in data]
        split = int(len(texts) * 0.8)
        return texts[:split], labels[:split], texts[split:], labels[split:]

    def run_all(self, model: QuantumNLPBase) -> dict:
        """Run model on all benchmarks, save JSON report."""
        results = {}

        for name, loader in [
            ("sentiment_sst2", lambda: self.sentiment_sst2(200)),
            ("custom_compositional", self.custom_compositional),
        ]:
            try:
                tr_x, tr_y, te_x, te_y = loader()
                if not tr_x:
                    continue
                model.fit(tr_x, tr_y, epochs=5, verbose=False)
                metrics = model.evaluate(te_x, te_y)
                results[name] = metrics
            except Exception as e:
                logger.warning(f"Benchmark {name} failed: {e}")
                results[name] = {"error": str(e)}

        report_path = "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark report saved to {report_path}")
        return results
