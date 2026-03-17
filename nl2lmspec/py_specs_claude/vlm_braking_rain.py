```python
import random
from scipy.stats import binom

# External callables assumed to be available in the environment
# Scenic(command: str) -> img
# L(img, question: str) -> str  (target VLM)
# J_VLM_QA(response: str, ground_truth: str) -> int

# ── Symbolic scene graph ──────────────────────────────────────────────────────
nu_ego     = "ego_vehicle"
nu_barrier = "road_barrier"
nu_empty   = "empty_path"
nu_rain    = "rain_weather"

edges = {
    (nu_ego, nu_barrier): "ahead",
    (nu_ego, nu_empty):   "ahead",
    (nu_ego, nu_rain):    "weather",
}

# ── Certification loop ────────────────────────────────────────────────────────
N_SAMPLES   = 200
CONFIDENCE  = 0.95
successes   = 0

for i in range(N_SAMPLES):
    # 1. Sample observable node uniformly from {ν_barrier, ν_empty}
    nu_obs = random.choice([nu_barrier, nu_empty])

    # 2. Retrieve relevant edges
    edge_ego_obs  = edges[(nu_ego, nu_obs)]
    edge_ego_rain = edges[(nu_ego, nu_rain)]

    # 3. Assemble symbolic subgraph g_sym (represented as a dict for clarity)
    g_sym = {
        "nodes": [nu_ego, nu_obs, nu_rain],
        "edges": [
            (nu_ego, nu_obs,  edge_ego_obs),
            (nu_ego, nu_rain, edge_ego_rain),
        ],
    }

    # 4. Form simulator command and render image via Scenic
    simulator_command = f"Setup {nu_ego} with {nu_obs} ahead in {nu_rain} weather"
    img = Scenic(simulator_command)

    # 5. Query the target VLM
    question = "Is it necessary for the vehicle to stop here?"
    r = L(img, question)

    # 6. Compute ground truth explicitly
    if nu_obs == nu_barrier:
        gt = "yes"
    else:
        gt = "no"

    # 7. Judge the response
    result = J_VLM_QA(r, gt)
    successes += result

    print(f"[{i+1:3d}/{N_SAMPLES}] nu_obs={nu_obs:<14s} gt={gt:<3s} "
          f"response={str(r):<60s} match={result}")

# ── Clopper-Pearson confidence interval ──────────────────────────────────────
alpha = 1.0 - CONFIDENCE
ci_low, ci_high = binom.interval(CONFIDENCE, N_SAMPLES, successes / N_SAMPLES)

# scipy binom.interval returns counts; convert to proportions
ci_low_prop  = binom.ppf(alpha / 2,     N_SAMPLES, successes / N_SAMPLES) / N_SAMPLES
ci_high_prop = binom.ppf(1 - alpha / 2, N_SAMPLES, successes / N_SAMPLES) / N_SAMPLES

# Use the standard Clopper-Pearson formula directly via binom.interval on proportion
from scipy.stats import beta as beta_dist

k = successes
n = N_SAMPLES
if k == 0:
    cp_low = 0.0
else:
    cp_low  = beta_dist.ppf(alpha / 2,     k,     n - k + 1)
if k == n:
    cp_high = 1.0
else:
    cp_high = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)

print("\n── Certification Result ─────────────────────────────────────────────────")
print(f"Samples        : {N_SAMPLES}")
print(f"Successes      : {successes}")
print(f"Empirical acc. : {successes / N_SAMPLES:.4f}")
print(f"Confidence     : {CONFIDENCE * 100:.0f}%")
print(f"Clopper-Pearson CI: [{cp_low:.4f}, {cp_high:.4f}]")
```