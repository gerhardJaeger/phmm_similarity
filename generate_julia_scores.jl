"""
Sample 1,000 ASJP string pairs from the lexibank pruned wordlist,
compute string_similarity for each using the Julia PHMM implementation,
and save (w1, w2, julia_score) to test_pairs.csv.

Requires the worldtree_msa repository at ../worldtree_msa (for the Julia
source files and lexibank wordlist data).  PHMM parameter files are
bundled in this repository.

  julia generate_julia_scores.jl
"""

using Pkg
Pkg.activate(@__DIR__)   # minimal env in phmm_similarity/

const WORLDTREE = joinpath(@__DIR__, "../worldtree_msa/code")
cd(WORLDTREE)

using ArgCheck
using CSV
using DataFrames
using Distributions
using JSON
using LinearAlgebra
using Random
using StatsFuns

include(joinpath(WORLDTREE, "phmm.jl"))
include(joinpath(WORLDTREE, "viterbi.jl"))

##

params = open(joinpath(@__DIR__, "phmm_parameters.json"), "r") do f
    JSON.parse(f)
end

lt = hcat(params["lt"]...)
lt[isnothing.(lt)] .= -Inf
lt = Matrix{Float64}(lt)
lp = Matrix{Float64}(hcat(params["lp"]...))
lq = Vector{Float64}(params["lq"])
η  = Float64(first(params["eta"]))

sound_df = CSV.File(joinpath(@__DIR__, "sound_probabilities.csv")) |> DataFrame
sounds = first.(sound_df.sounds)
sound_probs = sound_df.soundProbabilities

p = levPhmm(sounds)
p.lt = lt
p.lp = lp
p.lq = lq

##

function string_to_indices(w; alphabet=sounds)
    [findfirst(==(c), alphabet) for c in w]
end

function string_similarity(a::String, b::String)
    v1 = collect(a)
    v2 = collect(b)
    n1 = string_to_indices(a)
    n2 = string_to_indices(b)
    ll = viterbi(v1, v2, p).logprob
    sig = StatsFuns.logistic(η)
    null_model  = sum(log.(sound_probs[n1])) - sum(log.(sound_probs[n2]))
    null_model += log(sig) + log(1 - sig) * length(a)
    null_model += log(sig) + log(1 - sig) * length(b)
    return ll - null_model
end

##

wl = CSV.File(joinpath(WORLDTREE, "../data/lexibank_pruned_wordlist.csv")) |> DataFrame
dropmissing!(wl, :ASJP)

# Keep only rows whose ASJP string is non-empty and uses only known sounds
valid = filter(x -> x.ASJP != "" && all(c ∈ sounds for c in x.ASJP), wl)
@info "Rows after filtering: $(nrow(valid)) / $(nrow(wl))"

##

Random.seed!(42)
n = 1000
idx1 = rand(1:nrow(valid), n)
idx2 = rand(1:nrow(valid), n)

w1s = valid.ASJP[idx1]
w2s = valid.ASJP[idx2]

@info "Computing Julia scores..."
julia_scores = [string_similarity(w1s[i], w2s[i]) for i in 1:n]

out = DataFrame(w1=w1s, w2=w2s, julia_score=julia_scores)

out_path = joinpath(@__DIR__, "test_pairs.csv")
CSV.write(out_path, out)
@info "Saved $n pairs to $out_path"
