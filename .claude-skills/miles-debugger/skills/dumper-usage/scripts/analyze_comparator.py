#!/usr/bin/env python3
"""Analyze comparator JSON output and produce a structured diagnosis.

Usage:
    python analyze_comparator.py <report.jsonl> [--threshold 0.001] [--top-k 10]

Output: structured diagnosis with:
  - Summary statistics
  - Failed tensors ranked by severity
  - Layer-by-layer drift analysis
  - Pattern detection (common bug signatures)
  - Suggested next steps
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_report(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze(records: list[dict], threshold: float, top_k: int) -> dict:
    summary = None
    comparisons = []
    skipped = []
    errors = []
    config = None

    for r in records:
        rtype = r.get("type", "")
        if rtype == "summary":
            summary = r
        elif rtype == "comparison_tensor":
            comparisons.append(r)
        elif rtype == "comparison_skip":
            skipped.append(r)
        elif rtype == "comparison_error":
            errors.append(r)
        elif rtype == "config":
            config = r

    # Separate passed and failed
    passed = []
    failed = []
    for c in comparisons:
        diff = c.get("diff", {})
        if diff.get("passed", True):
            passed.append(c)
        else:
            failed.append(c)

    # Sort failed by rel_diff descending
    failed.sort(key=lambda c: c.get("diff", {}).get("rel_diff", 0), reverse=True)

    # Layer-by-layer analysis
    layer_diffs = defaultdict(list)
    for c in comparisons:
        layer_id = c.get("layer_id", c.get("meta", {}).get("layer_id", "unknown"))
        name = c.get("name", "unknown")
        rel_diff = c.get("diff", {}).get("rel_diff", 0)
        layer_diffs[layer_id].append({"name": name, "rel_diff": rel_diff})

    # Drift pattern detection
    layer_avg_diffs = {}
    for layer_id, diffs in sorted(layer_diffs.items(), key=lambda x: str(x[0])):
        avg = sum(d["rel_diff"] for d in diffs) / len(diffs) if diffs else 0
        layer_avg_diffs[layer_id] = avg

    # Pattern detection
    patterns = detect_patterns(failed, passed, skipped, layer_avg_diffs, threshold)

    return {
        "summary": {
            "total": summary.get("total", len(comparisons)) if summary else len(comparisons),
            "passed": summary.get("passed", len(passed)) if summary else len(passed),
            "failed": summary.get("failed", len(failed)) if summary else len(failed),
            "skipped": len(skipped),
            "errors": len(errors),
        },
        "top_failures": [
            {
                "name": c.get("name"),
                "rel_diff": c.get("diff", {}).get("rel_diff"),
                "max_abs_diff": c.get("diff", {}).get("max_abs_diff"),
                "layer_id": c.get("layer_id", c.get("meta", {}).get("layer_id")),
                "shape": c.get("shape"),
            }
            for c in failed[:top_k]
        ],
        "layer_drift": {
            str(k): round(v, 6) for k, v in sorted(layer_avg_diffs.items(), key=lambda x: str(x[0]))
        },
        "patterns_detected": patterns,
        "skipped_tensors": [{"name": s.get("name"), "reason": s.get("reason")} for s in skipped[:5]],
    }


def detect_patterns(
    failed: list[dict],
    passed: list[dict],
    skipped: list[dict],
    layer_avg_diffs: dict,
    threshold: float,
) -> list[dict]:
    patterns = []

    if not failed and not skipped:
        patterns.append({
            "pattern": "ALL_PASSED",
            "description": "All tensor comparisons passed. No issues detected.",
            "severity": "info",
        })
        return patterns

    # Pattern 1: Increasing drift across layers
    sorted_layers = sorted(
        [(k, v) for k, v in layer_avg_diffs.items() if k != "unknown"],
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999,
    )
    if len(sorted_layers) >= 3:
        diffs = [v for _, v in sorted_layers]
        if all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1)):
            patterns.append({
                "pattern": "MONOTONIC_LAYER_DRIFT",
                "description": f"Rel_diff increases monotonically across layers "
                               f"(layer 0: {diffs[0]:.6f} → layer {sorted_layers[-1][0]}: {diffs[-1]:.6f}). "
                               f"This is expected BF16 accumulation drift, not a bug.",
                "severity": "info",
                "action": f"Relax --diff-threshold to {max(diffs[-1] * 1.2, threshold):.4f}",
            })

    # Pattern 2: Attention diverges but layer_input matches
    failed_names = {c.get("name") for c in failed}
    passed_names = {c.get("name") for c in passed}
    if "layer_input" in passed_names and any("attn" in n for n in failed_names):
        patterns.append({
            "pattern": "ATTENTION_DIVERGENCE",
            "description": "layer_input passes but attention tensors fail. "
                          "Bug likely in attention computation (RoPE, softmax precision, "
                          "flash attention implementation, or causal mask).",
            "severity": "high",
            "action": "Compare Q/K/V before and after RoPE. Check attention backend and precision settings.",
        })

    # Pattern 3: MoE routing flip
    if "moe_topk_ids" in failed_names and "moe_router_logits" in passed_names:
        patterns.append({
            "pattern": "MOE_ROUTING_FLIP",
            "description": "Router logits match but topk_ids differ. "
                          "Small numerical differences flipped the expert selection. "
                          "This is a precision issue, not a logic bug.",
            "severity": "low",
            "action": "Enable R3 (--use-rollout-routing-replay) to force identical routing.",
        })

    # Pattern 4: All failures in a single layer
    failed_layers = {c.get("layer_id", c.get("meta", {}).get("layer_id")) for c in failed}
    if len(failed_layers) == 1 and None not in failed_layers:
        layer = failed_layers.pop()
        patterns.append({
            "pattern": "SINGLE_LAYER_FAILURE",
            "description": f"All failures are in layer {layer}. "
                          f"This suggests a layer-specific bug (e.g., gate, normalization, or weight loading).",
            "severity": "high",
            "action": f"Inspect layer {layer} code path. Dump more internal tensors in that layer.",
        })

    # Pattern 5: Everything fails (major mismatch)
    if failed and not passed:
        patterns.append({
            "pattern": "TOTAL_MISMATCH",
            "description": "ALL tensors fail comparison. "
                          "This usually means a fundamental issue: wrong weights, wrong input data, "
                          "or completely different model configurations.",
            "severity": "critical",
            "action": "Verify weights are loaded correctly. Check input_ids match between baseline and target. "
                     "Verify model config (hidden_size, num_heads, etc.) is identical.",
        })

    # Pattern 6: Only MLP fails
    if any("mlp" in n for n in failed_names) and all("mlp" in n or "moe" in n for n in failed_names):
        patterns.append({
            "pattern": "MLP_ONLY_FAILURE",
            "description": "Only MLP/MoE tensors fail. Attention tensors pass. "
                          "Bug is likely in MLP weight loading, activation function, or MoE dispatch.",
            "severity": "high",
            "action": "Check MLP weight shapes and TP slicing. Verify gate/up/down projection order.",
        })

    if not patterns:
        patterns.append({
            "pattern": "MIXED_FAILURES",
            "description": f"{len(failed)} tensors failed out of {len(failed) + len(passed)}. "
                          f"Review the top failures for the root cause.",
            "severity": "medium",
            "action": "Focus on the FIRST tensor that fails (earliest in the model). "
                     "That's usually closest to the bug.",
        })

    return patterns


def print_diagnosis(analysis: dict) -> None:
    s = analysis["summary"]
    print("=" * 60)
    print("DUMPER COMPARATOR ANALYSIS")
    print("=" * 60)
    print(f"\nSummary: {s['passed']} passed | {s['failed']} failed | "
          f"{s['skipped']} skipped | {s['errors']} errors | {s['total']} total")

    if analysis["top_failures"]:
        print(f"\nTop {len(analysis['top_failures'])} Failures:")
        for i, f in enumerate(analysis["top_failures"]):
            print(f"  {i+1}. {f['name']} (layer={f['layer_id']}): "
                  f"rel_diff={f['rel_diff']:.6f} max_abs={f['max_abs_diff']:.6f}")

    if analysis["layer_drift"]:
        print("\nLayer Drift (avg rel_diff):")
        for layer, diff in analysis["layer_drift"].items():
            bar = "#" * min(50, int(diff * 10000))
            print(f"  Layer {layer:>4s}: {diff:.6f} {bar}")

    print("\nPatterns Detected:")
    for p in analysis["patterns_detected"]:
        severity_icon = {"info": "[i]", "low": "[L]", "medium": "[M]", "high": "[H]", "critical": "[!]"}
        print(f"  {severity_icon.get(p['severity'], '[?]')} {p['pattern']}: {p['description']}")
        if "action" in p:
            print(f"      Action: {p['action']}")

    if analysis["skipped_tensors"]:
        print(f"\nSkipped Tensors (showing {len(analysis['skipped_tensors'])}):")
        for s in analysis["skipped_tensors"]:
            print(f"  - {s['name']}: {s['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze comparator JSON output")
    parser.add_argument("report", type=str, help="Path to comparator_report.jsonl")
    parser.add_argument("--threshold", type=float, default=0.001, help="Diff threshold")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top failures to show")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    args = parser.parse_args()

    if not Path(args.report).exists():
        print(f"Error: {args.report} not found", file=sys.stderr)
        sys.exit(1)

    records = load_report(args.report)
    analysis = analyze(records, args.threshold, args.top_k)

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_diagnosis(analysis)


if __name__ == "__main__":
    main()
