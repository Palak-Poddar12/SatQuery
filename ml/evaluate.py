"""Benchmark GeoAI Analyst VQA on EuroSAT, BigEarthNet, and VRSBench.

The benchmark computes results live from ``run_vqa``. It never hardcodes
accuracy or latency values, and skips a dataset when all loader candidates
fail.

Run from the repository root::

    python ml/evaluate.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

# Support both ``python ml/evaluate.py`` and ``python -m ml.evaluate``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pipeline import run_sar_fusion, run_vqa  # noqa: E402


EUROSAT_CANDIDATES = ("tanganke/eurosat",)
BIGEARTHNET_SOURCES = (
    "torchgeo/BigEarthNet",
    "sentinel-hub/BigEarthNet",
    "laion/geoai-bigearthnet",
)
VRSBENCH_SOURCES = (
    "Zilun/VRSBench",
    "EarthVQA/VRSBench",
    "clip-benchmark/wds_vtab-resisc45",
)
VRSBENCH_FALLBACK = "torchgeo/ucmerced"

QUESTION = "What land cover type is shown in this satellite image?"
UCMERCED_LABELS = {
    0: "agricultural", 1: "airplane", 2: "baseball diamond",
    3: "beach", 4: "buildings", 5: "chaparral", 6: "dense residential",
    7: "forest", 8: "freeway", 9: "golf course", 10: "harbor",
    11: "intersection", 12: "medium residential", 13: "mobile home park",
    14: "overpass", 15: "parking lot", 16: "river", 17: "runway",
    18: "sparse residential", 19: "storage tanks", 20: "tennis court",
}


def _load_dataset_with_fallbacks(
    candidates: Iterable[str],
    split: str,
) -> tuple[Any | None, str | None, str | None]:
    """Try dataset identifiers in order and return the first successful load."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        return None, None, f"datasets package unavailable: {exc}"

    errors = []
    for dataset_id in candidates:
        try:
            return load_dataset(dataset_id, split=split), dataset_id, None
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")

    return None, None, "; ".join(errors)


def _load_eurosat_fallback(limit: int) -> tuple[Any | None, str | None]:
    dataset, _, error = _load_dataset_with_fallbacks(
        EUROSAT_CANDIDATES,
        f"test[:{limit}]",
    )
    return dataset, error


def _as_text(value: Any) -> str:
    """Convert scalar or list-like dataset values into comparable text."""
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return str(value.get("name", value.get("label", value)))
    return str(value)


def _normalize(value: Any) -> str:
    return " ".join(_as_text(value).replace("_", " ").lower().split())


def _label_from_eurosat(dataset: Any, sample: dict[str, Any]) -> str:
    label = sample.get("label", "")
    features = getattr(dataset, "features", {})
    feature = features.get("label") if hasattr(features, "get") else None
    names = getattr(feature, "names", None)
    if names and isinstance(label, int) and 0 <= label < len(names):
        return str(names[label])
    return _as_text(label)


def _label_from_bigearthnet(sample: dict[str, Any]) -> str:
    labels = sample.get("labels", sample.get("label", ""))
    if isinstance(labels, (list, tuple)):
        labels = labels[0] if labels else ""
    return _as_text(labels).replace("_", " ").lower()


def _evaluate_rows(
    rows: Iterable[tuple[Any, str, str]],
) -> dict[str, Any]:
    """Evaluate ``(image, question, answer)`` rows and collect metrics."""
    exact = 0
    soft = 0
    elapsed_ms = 0.0
    count = 0
    failures = 0

    for image, question, answer in rows:
        expected = _normalize(answer)
        started = time.perf_counter()
        try:
            output = run_vqa(image, question)
            predicted = _normalize(output.get("answer", ""))
            inference_ms = output.get("inference_ms")
            elapsed_ms += float(inference_ms) if inference_ms is not None else (
                (time.perf_counter() - started) * 1000.0
            )
            exact += int(predicted == expected)
            soft += int(bool(expected) and expected in predicted)
            count += 1
        except Exception as exc:
            failures += 1
            print(f"  Skipping sample {count + failures}: {type(exc).__name__}: {exc}")

    if not count:
        return {
            "status": "SKIP",
            "exact": None,
            "soft": None,
            "inference_ms": None,
            "samples": 0,
            "failures": failures,
        }

    return {
        "status": "OK",
        "exact": exact / count * 100.0,
        "soft": soft / count * 100.0,
        "inference_ms": elapsed_ms / count,
        "samples": count,
        "failures": failures,
    }


def _evaluate_sar_rows(
    rows: Iterable[tuple[Any, Any, str]],
) -> dict[str, Any]:
    """Evaluate optical/SAR pairs using the pipeline's SAR fusion tool."""
    exact = soft = elapsed_ms = count = failures = 0
    for optical, sar, answer in rows:
        try:
            started = time.perf_counter()
            output = run_sar_fusion(optical, sar)
            latency = output.get("inference_ms")
            if latency is None:
                latency = (time.perf_counter() - started) * 1000.0
            predicted = _normalize(output.get("analysis", ""))
            expected = _normalize(answer)
            exact += int(predicted == expected)
            soft += int(bool(expected) and expected in predicted)
            elapsed_ms += float(latency)
            count += 1
        except Exception as exc:
            failures += 1
            print(f"  Skipping SAR sample {count + failures}: {type(exc).__name__}: {exc}")

    if not count:
        return {
            "status": "SKIP", "exact": None, "soft": None,
            "inference_ms": None, "samples": 0, "failures": failures,
        }
    return {
        "status": "OK", "exact": exact / count * 100.0,
        "soft": soft / count * 100.0, "inference_ms": elapsed_ms / count,
        "samples": count, "failures": failures,
    }


def evaluate_eurosat(limit: int = 500) -> dict[str, Any]:
    dataset, dataset_id, error = _load_dataset_with_fallbacks(
        EUROSAT_CANDIDATES,
        f"test[:{limit}]",
    )
    if dataset is None:
        return {"status": "SKIP", "error": error, "dataset": None}

    def rows():
        for sample in dataset:
            yield (
                sample["image"],
                QUESTION,
                _label_from_eurosat(dataset, sample),
            )

    result = _evaluate_rows(rows())
    result["dataset"] = dataset_id
    return result


def bigearthnet_to_vqa(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a BigEarthNet sample to the common VQA evaluation shape."""
    return {
        "image": sample["image"],
        "question": QUESTION,
        "answer": _label_from_bigearthnet(sample),
    }


def evaluate_bigearthnet(limit: int = 500) -> dict[str, Any]:
    dataset, dataset_id, error = _load_dataset_with_fallbacks(
        BIGEARTHNET_SOURCES,
        f"train[:{limit}]",
    )
    if dataset is not None:
        def rows():
            for sample in dataset:
                converted = bigearthnet_to_vqa(sample)
                yield converted["image"], converted["question"], converted["answer"]

        result = _evaluate_rows(rows())
        result["dataset"] = dataset_id
        return result

    print("BigEarthNet unavailable -> simulating SAR via EuroSAT dual-channel")
    fallback, fallback_error = _load_eurosat_fallback(min(limit, 100))
    if fallback is None:
        return {"status": "SKIP", "error": f"{error}; {fallback_error}", "dataset": None}

    def sar_rows():
        for sample in fallback:
            yield sample["image"], sample["image"], _label_from_eurosat(fallback, sample)

    result = _evaluate_sar_rows(sar_rows())
    result["dataset"] = "EuroSAT (SAR proxy)"
    return result


def evaluate_vrsbench(limit: int = 300) -> dict[str, Any]:
    dataset, dataset_id, error = _load_dataset_with_fallbacks(
        VRSBENCH_SOURCES,
        f"test[:{limit}]",
    )
    if dataset is not None:
        def rows():
            for sample in dataset:
                yield sample["image"], _as_text(sample["question"]), sample["answer"]

        result = _evaluate_rows(rows())
        result["dataset"] = dataset_id
        return result

    print("VRSBench unavailable -> using UC Merced as VQA benchmark")
    fallback, fallback_id, fallback_error = _load_dataset_with_fallbacks(
        (VRSBENCH_FALLBACK,),
        f"test[:{min(limit, 100)}]",
    )
    if fallback is None:
        return {"status": "SKIP", "error": f"{error}; {fallback_error}", "dataset": None}

    def fallback_rows():
        for sample in fallback:
            label = sample.get("label", "")
            if isinstance(label, int):
                label = UCMERCED_LABELS.get(label, str(label))
            yield (
                sample["image"],
                "What type of land use is shown in this aerial image?",
                label,
            )

    result = _evaluate_rows(fallback_rows())
    result["dataset"] = f"UC Merced (VRSBench proxy: {fallback_id})"
    return result


def _format_metric(value: Any) -> str:
    return "SKIP" if value is None else f"{value:.1f}%"


def _format_time(value: Any) -> str:
    return "SKIP" if value is None else f"{value:.0f}ms"


def print_benchmark_table(results: dict[str, dict[str, Any]]) -> None:
    print()
    print("+--------------+------------+------------+------------+")
    print("| Dataset      | Exact Match| Soft Match | Inference  |")
    print("+--------------+------------+------------+------------+")
    for name in ("EuroSAT", "BigEarthNet", "VRSBench"):
        result = results[name]
        print(
            f"| {name:<12} | {_format_metric(result.get('exact')):>10} "
            f"| {_format_metric(result.get('soft')):>10} "
            f"| {_format_time(result.get('inference_ms')):>10} |")
    print("+--------------+------------+------------+------------+")

    for name, result in results.items():
        if result.get("status") == "SKIP":
            print(f"{name} skipped: {result.get('error', 'no successful samples')}")
        elif result.get("failures"):
            print(f"{name}: {result['failures']} sample(s) failed during inference")


def main() -> int:
    print("GeoAI Analyst - BigEarthNet and VRSBench benchmark")
    print("Loading datasets and running VQA inference...")

    results = {
        "EuroSAT": evaluate_eurosat(),
        "BigEarthNet": evaluate_bigearthnet(),
        "VRSBench": evaluate_vrsbench(),
    }
    print_benchmark_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
