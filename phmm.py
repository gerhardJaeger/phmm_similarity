"""PHMM-based string similarity for ASJP-transcribed word forms.

Re-implementation of string_similarity() from phmm_dist.jl in the
worldtree_msa project.

The Pair Hidden Markov Model (PHMM) has three emitting states:
  M  — match:   consumes one symbol from each string
  X  — gap in w2: consumes one symbol from w1 only
  Y  — gap in w1: consumes one symbol from w2 only

All probability parameters are stored log-transformed.

Transition matrix lt has shape (4, 5):
  rows: {start, M, X, Y}
  cols: {unused, M, X, Y, End}

Emission matrix lp has shape (n, n), symmetric, for the M state.
Emission vector lq has shape (n,) for the X and Y states.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Phmm:
    alphabet: list[str]
    lt: np.ndarray   # (4, 5) log-transition matrix
    lp: np.ndarray   # (n, n) log-emission for match state
    lq: np.ndarray   # (n,)   log-emission for gap states


def _parse_julia_matrix(data: list) -> np.ndarray:
    """Parse a matrix serialized by Julia's JSON.jl (stored column-major).

    Julia writes an (r, c) matrix as a JSON array of c columns, each of
    length r.  None encodes -Inf for forbidden transitions.
    """
    cols = [
        [float("-inf") if x is None else float(x) for x in col]
        for col in data
    ]
    return np.array(cols, dtype=float).T  # shape: (r, c)


def load_phmm(params_path: str | Path, sounds_path: str | Path) -> tuple:
    """Load PHMM parameters from the worldtree_msa output files.

    Parameters
    ----------
    params_path : path to phmm_parameters.json
    sounds_path : path to sound_probabilities.csv

    Returns
    -------
    phmm            : Phmm dataclass
    sound_probs     : 1-D ndarray of unigram symbol probabilities
    eta             : scalar float, geometric-length null-model parameter
    """
    with open(params_path) as f:
        params = json.load(f)

    lt = _parse_julia_matrix(params["lt"])
    lp = _parse_julia_matrix(params["lp"])
    lq = np.array(params["lq"], dtype=float)

    eta_raw = params["eta"]
    eta = float(eta_raw[0]) if isinstance(eta_raw, list) else float(eta_raw)

    # Julia Chars serialize as single-character strings
    alphabet = [s[0] for s in params["alphabet"]]

    with open(sounds_path, newline="") as f:
        rows = list(csv.DictReader(f))

    sounds = [row["sounds"][0] for row in rows]
    sound_probs = np.array([float(row["soundProbabilities"]) for row in rows])

    # Sanity check: alphabet in params must match sounds file
    assert alphabet == sounds, (
        "Alphabet mismatch between phmm_parameters.json and sound_probabilities.csv"
    )

    return Phmm(alphabet=alphabet, lt=lt, lp=lp, lq=lq), sound_probs, eta


def viterbi(w1: str, w2: str, p: Phmm) -> float:
    """Return the log-probability of the most likely PHMM alignment of w1 and w2.

    Mirrors viterbi() in viterbi.jl.  The dp array has shape
    (n+1, m+1, 3) where the three slices correspond to states M, X, Y.
    """
    idx = {c: i for i, c in enumerate(p.alphabet)}
    v1 = [idx[c] for c in w1]
    v2 = [idx[c] for c in w2]
    n, m = len(w1), len(w2)

    dp = np.full((n + 1, m + 1, 3), -np.inf)

    # Initial transitions from the start state (lt row 0)
    if n >= 1 and m >= 1:
        dp[1, 1, 0] = p.lt[0, 1] + p.lp[v1[0], v2[0]]   # start → M
    if n >= 1:
        dp[1, 0, 1] = p.lt[0, 2] + p.lq[v1[0]]           # start → X
    if m >= 1:
        dp[0, 1, 2] = p.lt[0, 3] + p.lq[v2[0]]           # start → Y

    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0 and j > 0 and (i, j) != (1, 1):
                # M: align w1[i-1] with w2[j-1]
                dp[i, j, 0] = (
                    np.max(dp[i - 1, j - 1, :] + p.lt[1:4, 1])
                    + p.lp[v1[i - 1], v2[j - 1]]
                )
            if i > 0 and (i, j) != (1, 0):
                # X: gap in w2, consume w1[i-1]
                dp[i, j, 1] = (
                    np.max(dp[i - 1, j, :] + p.lt[1:4, 2])
                    + p.lq[v1[i - 1]]
                )
            if j > 0 and (i, j) != (0, 1):
                # Y: gap in w1, consume w2[j-1]
                dp[i, j, 2] = (
                    np.max(dp[i, j - 1, :] + p.lt[1:4, 3])
                    + p.lq[v2[j - 1]]
                )

    return float(np.max(dp[n, m, :] + p.lt[1:4, 4]))  # end transitions


def string_similarity(
    a: str,
    b: str,
    p: Phmm,
    sound_probs: np.ndarray,
    eta: float,
) -> float:
    """PHMM log-likelihood-ratio similarity between ASJP strings a and b.

    Mirrors string_similarity() in phmm_dist.jl:

        similarity = log P(a, b | PHMM) - log P(a, b | null)

    The null model generates each string independently: symbols drawn
    from unigram frequencies (sound_probs), lengths drawn from a
    geometric distribution with stop probability sigmoid(eta).

    Note: the null model subtracts the log-probability of b rather than
    adding it (consistent with phmm_dist.jl and optimize_phmm.jl).
    """
    idx = {c: i for i, c in enumerate(p.alphabet)}
    n1 = [idx[c] for c in a]
    n2 = [idx[c] for c in b]

    ll = viterbi(a, b, p)

    sig = 1.0 / (1.0 + np.exp(-eta))
    null_model = (
        np.sum(np.log(sound_probs[n1])) - np.sum(np.log(sound_probs[n2]))
        + np.log(sig) + np.log(1.0 - sig) * len(a)
        + np.log(sig) + np.log(1.0 - sig) * len(b)
    )
    return ll - null_model
