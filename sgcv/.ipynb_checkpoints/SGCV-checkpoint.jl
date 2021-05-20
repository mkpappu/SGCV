module SGCV

using ForneyLab
using HCubature
using FastGaussQuadrature
using ForwardDiff
using LinearAlgebra

include("switching_gaussian_controlled_variance.jl")
include("rules_prototypes.jl")
include("update_rules.jl")

end  # module ARnode
