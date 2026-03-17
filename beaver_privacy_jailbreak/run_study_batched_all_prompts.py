from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_SCRIPT_DIR = Path(__file__).resolve().parent
_LLMCERT_ROOT = _SCRIPT_DIR.parent
_BEAVER_ROOT = _LLMCERT_ROOT.parent / "Beaver"

for _p in [str(_LLMCERT_ROOT), str(_BEAVER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import beaver  # noqa: E402
from experiments.enron.enron import check_call_fn, constraint_fn, load_prompts  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from certification.aggregate_bounds import empirical_bernstein_bounds  # noqa: E402
from certification.privacy_jailbreaks import gen_jb  # noqa: E402

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]
ALL_PROMPT_IDS = list(range(10))
N_JAILBREAKS = 50
DEFAULT_OUTPUT = _LLMCERT_ROOT / "output" / "enron_study_results_batched_all_prompts.csv"
DEFAULT_INSTANCE_OUTPUT = (
    _LLMCERT_ROOT / "output" / "enron_study_results_batched_all_prompts_instances.csv"
)

SUMMARY_FIELDS = ["model", "prompt_idx", "certified_safe_lb", "certified_safe_ub"]
INSTANCE_FIELDS = ["model", "instance_idx", "prompt_idx", "jb_idx", "lb", "ub", "status", "error"]


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def extract_bounds(r: dict) -> tuple[float | None, float | None]:
    if "complete_prob" in r and "incomplete_prob" in r:
        return r["complete_prob"], r["complete_prob"] + r["incomplete_prob"]
    if "lower_bound" in r and "upper_bound" in r:
        return r["lower_bound"], r["upper_bound"]
    return None, None


def run_model(model: str, prompt_ids: list[int], server_addr: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = load_prompts(debug_ids=prompt_ids)

    instances = []
    idx_to_meta = {}
    idx = 0
    for prompt in prompts:
        pidx = int(prompt["idx"])
        base_question = prompt["prompt"]
        for jb_idx in range(N_JAILBREAKS):
            inst = dict(prompt)
            inst["prompt"] = gen_jb(tokenizer) + " " + base_question
            inst["idx"] = idx
            instances.append(inst)
            idx_to_meta[idx] = (pidx, jb_idx)
            idx += 1

    run_kwargs = {
        "prompts": instances,
        "constraint_fn": constraint_fn,
        "check_call_fn": check_call_fn,
        "model": model,
        "epsilon": 0.1,
    }
    if server_addr:
        run_kwargs["server_addr"] = server_addr

    expected = len(instances)
    print(f"\nModel: {model}")
    print(f"  Prepared {expected} instances ({len(prompts)} prompts x {N_JAILBREAKS} jailbreaks)")

    results = beaver.run(**run_kwargs)
    
    by_idx = {int(r["idx"]): r for r in results if "idx" in r}
    print(f"  Beaver returned {len(results)} results (expected {expected})")

    grouped = defaultdict(lambda: {"lowers": [], "uppers": []})
    instance_rows = []
    success = 0
    for i in range(expected):
        prompt_idx, jb_idx = idx_to_meta[i]
        r = by_idx.get(i)
        if r is None:
            instance_rows.append(
                {"model": model, "instance_idx": i, "prompt_idx": prompt_idx, "jb_idx": jb_idx, "lb": "", "ub": "", "status": "missing", "error": "missing result"}
            )
            continue
        if "error" in r:
            instance_rows.append(
                {"model": model, "instance_idx": i, "prompt_idx": prompt_idx, "jb_idx": jb_idx, "lb": "", "ub": "", "status": "error", "error": r.get("error", "")}
            )
            continue
        lb, ub = extract_bounds(r)
        if lb is None:
            instance_rows.append(
                {"model": model, "instance_idx": i, "prompt_idx": prompt_idx, "jb_idx": jb_idx, "lb": "", "ub": "", "status": "invalid", "error": f"unexpected keys: {sorted(r.keys())}"}
            )
            continue
        grouped[prompt_idx]["lowers"].append(lb)
        grouped[prompt_idx]["uppers"].append(ub)
        instance_rows.append(
            {"model": model, "instance_idx": i, "prompt_idx": prompt_idx, "jb_idx": jb_idx, "lb": round(lb, 10), "ub": round(ub, 10), "status": "ok", "error": ""}
        )
        success += 1

    print(f"  Successful instance bounds: {success}/{expected}")

    summary_rows = []
    for prompt in prompts:
        pidx = int(prompt["idx"])
        lowers = grouped[pidx]["lowers"]
        uppers = grouped[pidx]["uppers"]
        if len(lowers) < 2:
            print(f"  Prompt {pidx}: insufficient successful bounds ({len(lowers)})")
            continue
        lb, ub = empirical_bernstein_bounds(np.array(lowers), np.array(uppers))
        summary_rows.append(
            {"model": model, "prompt_idx": pidx, "certified_safe_lb": round(lb, 6), "certified_safe_ub": round(ub, 6)}
        )
        print(f"  Prompt {pidx}: certified safe=[{lb:.4f}, {ub:.4f}] ({len(lowers)} bounds)")

    return summary_rows, instance_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Model to run")
    p.add_argument("--prompts", help="Comma-separated prompt indices")
    p.add_argument("--server_addr", help="vLLM server URL (skip server management)")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--instance_output", default=str(DEFAULT_INSTANCE_OUTPUT))
    args = p.parse_args()

    models = [args.model] if args.model else MODELS
    prompt_ids = [int(x) for x in args.prompts.split(",")] if args.prompts else ALL_PROMPT_IDS

    all_summary_rows = []
    all_instance_rows = []
    for model in models:
        summary_rows, instance_rows = run_model(model, prompt_ids, args.server_addr)
        all_summary_rows.extend(summary_rows)
        all_instance_rows.extend(instance_rows)

    write_csv(args.output, SUMMARY_FIELDS, all_summary_rows)
    write_csv(args.instance_output, INSTANCE_FIELDS, all_instance_rows)
    print(f"\n{len(all_summary_rows)} rows written to {args.output}")
    print(f"{len(all_instance_rows)} rows written to {args.instance_output}")

    # ── Comparison summary CSV ────────────────────────────────────────────────
    import statistics as _stats
    comparison_rows = []
    for model in models:
        model_rows = [r for r in all_summary_rows if r["model"] == model]
        lbs = [r["certified_safe_lb"] for r in model_rows]
        ubs = [r["certified_safe_ub"] for r in model_rows]
        # per-prompt rows
        for r in model_rows:
            comparison_rows.append({
                "model": r["model"],
                "prompt_idx": r["prompt_idx"],
                "type": "per_prompt",
                "certified_safe": f"[{r['certified_safe_lb']:.6f}, {r['certified_safe_ub']:.6f}]",
                "lb": r["certified_safe_lb"],
                "ub": r["certified_safe_ub"],
            })
        # aggregate rows
        if lbs:
            for agg, fn in [("median", _stats.median), ("average", lambda x: sum(x)/len(x))]:
                comparison_rows.append({
                    "model": model,
                    "prompt_idx": agg,
                    "type": agg,
                    "certified_safe": f"[{fn(lbs):.6f}, {fn(ubs):.6f}]",
                    "lb": round(fn(lbs), 6),
                    "ub": round(fn(ubs), 6),
                })

    comparison_path = str(Path(args.output).parent / "comparison_summary_with_jb.csv")
    write_csv(comparison_path, ["model", "prompt_idx", "type", "certified_safe", "lb", "ub"], comparison_rows)
    print(f"\nComparison summary written to {comparison_path}")
    print()
    for row in comparison_rows:
        if row["type"] in ("median", "average"):
            print(f"  {row['model'].split('/')[-1]:30s}  {row['type']:8s}  {row['certified_safe']}")


if __name__ == "__main__":
    main()
