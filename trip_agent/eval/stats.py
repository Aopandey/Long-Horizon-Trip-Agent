"""Aggregates and paired comparisons across variants."""

from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np

SUMMARY_METRICS = (
    "valid",
    "fraction_satisfied",
    "false_reliance_items",
    "commitments_on_false_info",
    "first_error_position",
    "errors_after_first",
    "recovery_success_rate",
    "verify_calls",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "wall_ms",
)


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(rows: list[dict], by=("condition", "horizon", "variant")) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in by)].append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        entry = dict(zip(by, key, strict=True))
        entry["n"] = len(members)
        for metric in SUMMARY_METRICS:
            entry[metric] = _mean(m.get(metric) for m in members)
        out.append(entry)
    return out


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Mean of (b - a) with a percentile bootstrap interval over pairs."""
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if diffs.size == 0:
        return {"n": 0, "mean_diff": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, diffs.size, size=(n_boot, diffs.size))].mean(axis=1)
    alpha = (1 - level) / 2
    return {
        "n": int(diffs.size),
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_low": round(float(np.quantile(means, alpha)), 4),
        "ci_high": round(float(np.quantile(means, 1 - alpha)), 4),
    }


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes."""
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "p_value": round(p, 6)}


def compare_variants(rows: list[dict], baseline: str = "standard", metric: str = "valid") -> list[dict]:
    """Pair each variant with the baseline on (scenario, condition, repeat), per condition and horizon."""
    index = {(r["variant"], r["scenario_id"], r["condition"], r["repeat"]): r for r in rows}
    variants = sorted({r["variant"] for r in rows} - {baseline})
    cells = sorted({(r["condition"], r["horizon"]) for r in rows})
    out = []
    for condition, horizon in cells:
        keys = sorted(
            {(r["scenario_id"], r["repeat"]) for r in rows if r["condition"] == condition and r["horizon"] == horizon}
        )
        for variant in variants:
            pairs = [
                (index[(baseline, s, condition, rep)], index[(variant, s, condition, rep)])
                for s, rep in keys
                if (baseline, s, condition, rep) in index and (variant, s, condition, rep) in index
            ]
            a = [p[0].get(metric) for p in pairs]
            b = [p[1].get(metric) for p in pairs]
            usable = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
            entry = {
                "condition": condition,
                "horizon": horizon,
                "variant": variant,
                "baseline": baseline,
                "metric": metric,
            }
            entry.update(paired_bootstrap([float(x) for x, _ in usable], [float(y) for _, y in usable]))
            if metric == "valid":
                entry.update(mcnemar_exact([bool(x) for x, _ in usable], [bool(y) for _, y in usable]))
            out.append(entry)
    return out
