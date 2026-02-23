"""
Load test_pairs.csv (w1, w2, julia_score) produced by generate_julia_scores.jl,
compute string_similarity with the Python PHMM implementation, and compare.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from phmm import load_phmm, string_similarity

here = Path(__file__).parent
worldtree = here.parent / "worldtree_msa" / "code"

p, sound_probs, eta = load_phmm(
    worldtree / "phmm_parameters.json",
    worldtree / "sound_probabilities.csv",
)

pairs = pd.read_csv(here / "test_pairs.csv", keep_default_na=False)

pairs["python_score"] = [
    string_similarity(row.w1, row.w2, p, sound_probs, eta)
    for _, row in pairs.iterrows()
]

pairs["diff"] = pairs["python_score"] - pairs["julia_score"]
pairs["abs_diff"] = pairs["diff"].abs()

print(pairs[["w1", "w2", "julia_score", "python_score", "diff"]].to_string(max_rows=20))
print()
print(f"Max absolute difference : {pairs['abs_diff'].max():.6e}")
print(f"Mean absolute difference: {pairs['abs_diff'].mean():.6e}")
print(f"Pairs with diff > 1e-6  : {(pairs['abs_diff'] > 1e-6).sum()}")
print(f"Pairs with diff > 1e-4  : {(pairs['abs_diff'] > 1e-4).sum()}")

worst = pairs.nlargest(5, "abs_diff")[["w1", "w2", "julia_score", "python_score", "abs_diff"]]
print("\n5 largest discrepancies:")
print(worst.to_string(index=False))
