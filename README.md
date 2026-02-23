# phmm_similarity

Python implementation of a Pair Hidden Markov Model (PHMM) similarity measure for ASJP-transcribed word forms, ported from the [worldtree_msa](https://github.com/gerhardJaeger/worldtree_msa) Julia codebase.

## What it does

`string_similarity(a, b)` computes a log-likelihood ratio between a trained PHMM and a null model:

```
similarity(a, b) = log P(a, b | PHMM) − log P(a, b | null)
```

The PHMM has three emitting states — match (M), gap in b (X), gap in a (Y) — and was trained on cognate pairs from the [Lexibank](https://lexibank.clld.org/) database. The null model generates each string independently from unigram sound frequencies with geometric lengths. The log-probability of the best alignment is computed via the Viterbi algorithm.

## Usage

```python
from phmm import load_phmm, string_similarity
from pathlib import Path

here = Path(__file__).parent
p, sound_probs, eta = load_phmm(here / "phmm_parameters.json", here / "sound_probabilities.csv")

score = string_similarity("hand", "hant", p, sound_probs, eta)
```

Inputs are ASJP strings — single characters per sound segment. The bundled `sound_probabilities.csv` lists the valid alphabet.

## Requirements

- Python ≥ 3.11
- numpy

## Files

| File | Description |
|---|---|
| `phmm.py` | `Phmm` dataclass, `load_phmm`, `viterbi`, `string_similarity` |
| `phmm_parameters.json` | Trained PHMM parameters (transition and emission matrices) |
| `sound_probabilities.csv` | ASJP alphabet with unigram frequencies |
| `compare.py` | Numerical comparison against the Julia implementation |
| `generate_julia_scores.jl` | Generates reference scores from the Julia implementation (requires worldtree_msa at `../worldtree_msa`) |
