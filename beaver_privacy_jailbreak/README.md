# Beaver Privacy Jailbreak Bundle

This repo contains:

- `privacy_jailbreaks.py`
- `run_study_batched_all_prompts.py`

## What These Files Do

### `privacy_jailbreaks.py`

Builds randomized jailbreak prefixes with:

- a base template (`MAIN_JB`),
- extra instructions (`SIDE_INSTRUCTIONS`),
- random crossover + token mutation in `gen_jb(...)`.

Each call to `gen_jb(tokenizer)` returns one jailbreak string.

### `run_study_batched_all_prompts.py`

Runs a Beaver-based privacy study over:

- models in `MODELS`,
- prompt IDs in `ALL_PROMPT_IDS`,
- `N_JAILBREAKS=50` jailbreak variants per prompt.

For each instance, it prepends a generated jailbreak to a base Enron prompt, runs Beaver verification, and writes:

- per-prompt certified bounds,
- per-instance bounds/status,
- an aggregate comparison summary.

## Important Dependency Note

This script expects the same project layout used in the original environment:

- Beaver codebase available at a sibling path (e.g., `../Beaver`)
- Enron experiment utilities from Beaver (`experiments.enron.enron`)
- `certification.aggregate_bounds` available on `PYTHONPATH`

So this bundle is primarily a code snapshot; run-time dependencies come from Beaver/LLMCert-style setup.

## Expected Outputs

By default, the script writes:

- `output/enron_study_results_batched_all_prompts.csv`
- `output/enron_study_results_batched_all_prompts_instances.csv`
- `output/comparison_summary_with_jb.csv`

## Example Run

```bash
python run_study_batched_all_prompts.py
```

With options:

```bash
python run_study_batched_all_prompts.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompts 3,4,5 \
  --server_addr http://localhost:8081
```
