from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

from openai import OpenAI
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_BEAVER_ROOT = _ROOT.parent / "Beaver"
for _p in [str(_ROOT), str(_BEAVER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from beaver.server import start_server, stop_server  # noqa: E402
from experiments.enron.enron import constraint_fn, load_prompts  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from certification.privacy_jailbreaks import gen_jb  # noqa: E402

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]
OUTPUT = _ROOT / "output" / "enron_clopper_pearson_without_jb.csv"
ALPHA = 0.05
MAX_TOKENS = 32
TEMPERATURE = 1.0
TOP_P = 0.99


def resolve_tokenizer_source(model: str) -> str:
    # In offline mode, resolve HF model ids to local cache snapshots.
    if Path(model).exists():
        return model
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        return model
    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_dir.exists():
        return model
    snaps = [p for p in cache_dir.iterdir() if p.is_dir()]
    if not snaps:
        return model
    snaps.sort(key=lambda p: p.stat().st_mtime)
    return str(snaps[-1])


def clopper_pearson(x: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    def _betacf(a: float, b: float, x_: float) -> float:
        # Continued fraction for incomplete beta (Numerical Recipes style).
        max_iter = 200
        eps = 3e-14
        fpmin = 1e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x_ / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x_ / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x_ / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    def _incbeta(a: float, b: float, x_: float) -> float:
        # Regularized incomplete beta I_x(a,b).
        if x_ <= 0.0:
            return 0.0
        if x_ >= 1.0:
            return 1.0
        ln_bt = (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x_)
            + b * math.log(1.0 - x_)
        )
        bt = math.exp(ln_bt)
        if x_ < (a + 1.0) / (a + b + 2.0):
            return bt * _betacf(a, b, x_) / a
        return 1.0 - bt * _betacf(b, a, 1.0 - x_) / b

    def beta_ppf(q: float, a: float, b: float) -> float:
        a_t = torch.tensor(float(a), dtype=torch.float64)
        b_t = torch.tensor(float(b), dtype=torch.float64)
        q_t = torch.tensor(float(q), dtype=torch.float64)
        dist = torch.distributions.Beta(a_t, b_t)
        try:
            return float(dist.icdf(q_t).item())
        except NotImplementedError:
            if hasattr(torch.special, "betainc"):
                lo = torch.tensor(0.0, dtype=torch.float64)
                hi = torch.tensor(1.0, dtype=torch.float64)
                for _ in range(80):
                    mid = (lo + hi) / 2
                    cdf_mid = torch.special.betainc(a_t, b_t, mid)
                    if cdf_mid < q_t:
                        lo = mid
                    else:
                        hi = mid
                return float(((lo + hi) / 2).item())

            # Final fallback: pure-python bisection on regularized incomplete beta.
            lo = 0.0
            hi = 1.0
            for _ in range(100):
                mid = (lo + hi) / 2.0
                cdf_mid = _incbeta(float(a), float(b), mid)
                if cdf_mid < q:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2.0

    if x == 0:
        low = 0.0
    else:
        low = beta_ppf(alpha / 2, x, n - x + 1)
    if x == n:
        high = 1.0
    else:
        high = beta_ppf(1 - alpha / 2, x + 1, n - x)
    return low, high


def run_for_model(model: str, samples: int, server_addr: str | None, use_jailbreak: bool) -> list[dict]:
    prompts = load_prompts(debug_ids=list(range(10)))
    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_source(model))

    proc = None
    if server_addr:
        addr = server_addr
    else:
        proc = start_server(model)
        addr = "http://localhost:8081"

    rows: list[dict] = []
    try:
        client = OpenAI(base_url=f"{addr.rstrip('/')}/v1", api_key="EMPTY")
        for p in prompts:
            prompt_idx = int(p["idx"])
            base_prompt = p["prompt"]
            unsafe = 0
            for _ in range(samples):
                prompt = base_prompt
                # if use_jailbreak:
                #     prompt = gen_jb(tokenizer) + " " + base_prompt
                resp = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                text = (resp.choices[0].text or "").strip()
                if not constraint_fn(p, text):
                    unsafe += 1

            low, high = clopper_pearson(unsafe, samples, ALPHA)
            rows.append(
                {
                    "model": model,
                    "prompt_idx": prompt_idx,
                    "samples": samples,
                    "unsafe_count": unsafe,
                    "unsafe_rate": round(unsafe / samples, 6),
                    "unsafe_cp_lb": round(low, 6),
                    "unsafe_cp_ub": round(high, 6),
                    "use_jailbreak": False,
                }
            )
            print(f"[{model}] prompt {prompt_idx}: unsafe={unsafe}/{samples} CP=[{low:.4f}, {high:.4f}]")
    finally:
        if proc is not None:
            stop_server(proc)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "prompt_idx",
                "samples",
                "unsafe_count",
                "unsafe_rate",
                "unsafe_cp_lb",
                "unsafe_cp_ub",
                "use_jailbreak",
            ],
        )
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Run one model; default runs both.")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--server_addr", help="Use existing vLLM server.")
    ap.add_argument("--no_jailbreak", action="store_true")
    ap.add_argument("-o", "--output", default=str(OUTPUT))
    args = ap.parse_args()

    models = [args.model] if args.model else MODELS
    if args.server_addr and len(models) > 1:
        raise ValueError("--server_addr requires --model (single model).")

    out = Path(args.output)
    total_rows = 0
    for m in models:
        model_rows = run_for_model(
            model=m,
            samples=args.samples,
            server_addr=args.server_addr,
            use_jailbreak=not args.no_jailbreak,
        )
        write_csv(out, model_rows)
        total_rows += len(model_rows)
        print(f"Wrote {len(model_rows)} rows for {m} to {out}")

    print(f"\nWrote {total_rows} total rows to {out}")


if __name__ == "__main__":
    main()
