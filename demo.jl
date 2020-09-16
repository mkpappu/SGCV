include("SGCV.jl")
using Revise
using Main.SGCV
using ForneyLab
using LinearAlgebra
using Plots
using ProgressMeter
using LinearAlgebra

n_samples = 99
function generate_swtiching_hgf(n_samples)
    κs = [0.1, 2.0]
    ωs = [1.0, -2.0]
    z = Vector{Float64}(undef, n_samples)
    x = Vector{Float64}(undef, n_samples)
    z[1] = 0.0
    x[1] = 0.0
    rw = 0.01
    std_x = []

    for i in 2:n_samples
        z[i] = z[i - 1] + sqrt(rw)*randn()
        if i < 25
            push!(std_x, sqrt(exp(κs[1]*z[i] + ωs[1])))
            x[i] = x[i - 1] + std_x[end]*randn()
        elseif 25 <= i < 50
            push!(std_x, sqrt(exp(κs[2]*z[i] + ωs[2])))
            x[i] = x[i - 1] + std_x[end]*randn()
        else
            push!(std_x, sqrt(exp(κs[1]*z[i] + ωs[1])))
            x[i] = x[i - 1] + std_x[end]*randn()
        end
    end
    return x, std_x
end

using JLD
d = load("dump.jld")
x, std_x, categories, observations = d["state"], d["variance"], d["category"], d["observations"]

pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,2,'0')) # Left-pads a number with zeros, converts it to symbol and appends to sym

function generate_algorithm(ndim, n_samples)
    fg = FactorGraph()
    z = Vector{Variable}(undef, n_samples)
    x = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)
    s = Vector{Variable}(undef, n_samples)
    @RV A ~ Dirichlet([1.00 1.00; 1.00 1.00])
    @RV κ ~ GaussianMeanPrecision([0.1, 2.0], [10.0 0.0; 0.0 10.0])
    @RV ω ~ GaussianMeanPrecision([1.0, -2.0], [10.0 0.0; 0.0 10.0])
    @RV z[1] ~ GaussianMeanPrecision(placeholder(:mz_prior1), placeholder(:wz_prior1))
    @RV x[1] ~ GaussianMeanPrecision(placeholder(:mx_prior1), placeholder(:wx_prior1))
    @RV y[1] ~ GaussianMeanPrecision(x[1], placeholder(:wy_prior1))
    @RV s[1] ~ ForneyLab.Categorical(ones(ndim) ./ ndim)
    placeholder(y[1], :y, index = 1)
    for t in 2:n_samples
        @RV s[t] ~ Transition(s[t-1], A)
        @RV z[t] ~ GaussianMeanPrecision(z[t - 1], placeholder(pad(:wz_transition, t)))
        @RV x[t] ~ SwitchingGaussianControlledVariance(x[t - 1], z[t],κ,ω,s[t])
        @RV y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end
    algo = variationalAlgorithm(x, z ,s, κ, ω, A, ids=[:X :Z :S :K :O :A],free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

code = generate_algorithm(2, n_samples)
eval(Meta.parse(code))

#x, std_x = generate_swtiching_hgf(n_samples)
plot(x)

ndims = 2
wy_prior1 = 1.0
κ_m_prior, κ_w_prior = [0.1, 2.0], [10.0 0.0; 0.0 10.0]
ω_m_prior, ω_w_prior = [1.0, -2.0], [10.0 0.0; 0.0 10.0]
z_m_prior, z_w_prior = 1.0, 1.0
x_m_prior, x_w_prior = 0.0, 0.1
x_x_m_prior, x_x_w_prior = zeros(2), 1*diageye(2)
z_z_m_prior, z_z_w_prior = zeros(2), 1*diageye(2)
z_w_transition_prior = 100.0
y_w_transition_prior = 1.0

marginals = Dict()
marginals[:A] = vague(Dirichlet, (2, 2))
marginals[:κ] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = κ_m_prior, w = κ_w_prior)
marginals[:ω] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = ω_m_prior, w = ω_w_prior)
for t = 1:n_samples
    marginals[Symbol(:z_,t)] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = z_m_prior, w = z_w_prior)
    marginals[Symbol(:x_,t)] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = x_m_prior, w = x_w_prior)
    marginals[Symbol(:s_,t)] = vague(ForneyLab.Categorical, ndims)
    marginals[Symbol(:z_,t)*:_*Symbol(:z_,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = z_z_m_prior, w = z_z_w_prior)
    marginals[Symbol(:x_,t)*:_*Symbol(:x_,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = x_x_m_prior, w = x_x_w_prior)
end
data = Dict()

data[:mz_prior1] = z_m_prior
data[:wz_prior1] = z_w_prior
data[:mx_prior1] = x_m_prior
data[:wx_prior1] = x_w_prior
data[:wy_prior1] = wy_prior1
for t = 1:n_samples
    data[pad(:wz_transition, t)] = z_w_transition_prior
    data[pad(:wy_transition, t)] = y_w_transition_prior
end
#obs = x .+ randn(length(x))
data[:y] = observations

n_its = 10
fe = Vector{Float64}(undef, n_its)

##
@showprogress "Iterations" for i = 1:n_its
    stepZ!(data, marginals)
    stepX!(data, marginals)
    stepS!(data, marginals)
    stepA!(data, marginals)
    stepO!(data, marginals)
    stepK!(data, marginals)
    fe[i] = freeEnergy(data, marginals)
end

mz = [ForneyLab.unsafeMean(marginals[Symbol(:z_,t)]) for t=1:n_samples]
vz = [ForneyLab.unsafeVar(marginals[Symbol(:z_,t)]) for t=1:n_samples]
mx = [ForneyLab.unsafeMean(marginals[Symbol(:x_,t)]) for t=1:n_samples]
vx = [ForneyLab.unsafeVar(marginals[Symbol(:x_,t)]) for t=1:n_samples]
ms = [ForneyLab.unsafeMean(marginals[Symbol(:s_,t)]) for t=1:n_samples]

plot(mx, ribbon=sqrt.(vx))
plot!(x)
scatter!(obs)

categories = [x[2] for x in findmax.(ms)]
scatter(categories)