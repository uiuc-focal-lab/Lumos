```python
import random
from scipy.stats import binom

def certify():
    # Load the graph
    G = getGraph("primekg.csv")
    
    n_samples = 1000
    confidence = 0.95
    successes = 0
    
    for _ in range(n_samples):
        # Sample a drug node d
        d = meas(G, "drug")
        
        # Sample dis1: a neighbour of d connected by "treat" edge
        dis1 = meas(G, d, "treat")
        
        # Sample dis2: a neighbour of d connected by "contraindicate" edge
        dis2 = meas(G, d, "contraindicate")
        
        # Find all drugs that treat dis1 AND are contraindicated for dis2
        correct_drugs = set(getAllDrugsTreatAndContraindicate(G, dis1, dis2))
        
        # Sample 3 distractor drug nodes, each unique and not in correct_drugs
        distractors = []
        while len(distractors) < 3:
            dist = meas(G, "drug")
            dist_name = getNodeName(dist)
            d_name = getNodeName(d)
            if dist_name != d_name and dist_name not in correct_drugs and dist_name not in [getNodeName(x) for x in distractors]:
                distractors.append(dist)
        
        dist1, dist2_node, dist3 = distractors[0], distractors[1], distractors[2]
        
        # Form ordered set {d, dist1, dist2, dist3}
        ordered_set = [d, dist1, dist2_node, dist3]
        
        # Shuffle by sampling without replacement using meas
        shuffled = []
        remaining = list(ordered_set)
        while len(remaining) > 0:
            chosen = meas(remaining)
            shuffled.append(chosen)
            remaining.remove(chosen)
        
        options = shuffled
        
        # Assign labels A), B), C), D)
        labels = ["A)", "B)", "C)", "D)"]
        
        # Find ground truth label (label assigned to d)
        ground_truth_label = None
        for i, node in enumerate(options):
            if getNodeName(node) == getNodeName(d):
                ground_truth_label = labels[i]
                break
        
        # Get context: last attribute of each of d, dis1, dis2 concatenated
        d_context = getLastAttribute(d)
        dis1_context = getLastAttribute(dis1)
        dis2_context = getLastAttribute(dis2)
        context = d_context + dis1_context + dis2_context
        
        # Form the question
        dis1_name = getNodeName(dis1)
        dis2_name = getNodeName(dis2)
        question = f"Which drug treats {dis1_name} and is contraindicated for {dis2_name}?"
        
        # Format options
        formatted_options = ""
        for i, node in enumerate(options):
            formatted_options += f"{labels[i]} {getNodeName(node)}\n"
        
        # Form the full prompt
        prompt = context + question + "\n" + formatted_options
        
        # Call the target LLM L
        response = L(prompt)
        
        # Judge the response
        result = J_QA(response, ground_truth_label)
        
        if result == 1:
            successes += 1
    
    # Clopper-Pearson confidence interval
    alpha = 1 - confidence
    lower, upper = binom.interval(confidence, n_samples, successes / n_samples)
    # Use the proper Clopper-Pearson method
    from scipy.stats import beta
    
    k = successes
    n = n_samples
    
    if k == 0:
        lower_cp = 0.0
    else:
        lower_cp = beta.ppf(alpha / 2, k, n - k + 1)
    
    if k == n:
        upper_cp = 1.0
    else:
        upper_cp = beta.ppf(1 - alpha / 2, k + 1, n - k)
    
    print(f"Certification Results (n={n_samples}, confidence={confidence*100}%)")
    print(f"Successes: {successes}/{n_samples}")
    print(f"Point estimate: {successes/n_samples:.4f}")
    print(f"Clopper-Pearson {confidence*100}% CI: [{lower_cp:.4f}, {upper_cp:.4f}]")
    
    return lower_cp, upper_cp

certify()
```