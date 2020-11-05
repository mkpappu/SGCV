include("sgcv/SGCV.jl")
using Revise
using Main.SGCV
using ForneyLab
using LinearAlgebra
using Plots
using ProgressMeter
using LinearAlgebra
using Random
#include("compatibility.jl")

Random.seed!(42)

n_samples = 100
switches = Array{Int64}(undef,n_samples)
switches[1:Int(round(n_samples/3))] .= 1;
switches[Int(round(n_samples/3))+1:2*Int(round(n_samples/3))] .= 2;
switches[2*Int(round(n_samples/3))+1:n_samples] .= 1;

function generate_swtiching_hgf(n_samples, switches)
    κs = [1.0, 1.0]
    ωs = [-2.0 2.0]
    z = Vector{Float64}(undef, n_samples)
    x = Vector{Float64}(undef, n_samples)
    z[1] = 0.0
    x[1] = 0.0
    rw = 0.01
    std_x = []

    for i in 2:n_samples
        z[i] = z[i - 1] + sqrt(rw)*randn()
        if switches[i] == 1
            push!(std_x, sqrt(exp(κs[1]*z[i] + ωs[1])))
            x[i] = x[i - 1] + std_x[end]*randn()
        elseif switches[i] == 2
            push!(std_x, sqrt(exp(κs[2]*z[i] + ωs[2])))
            x[i] = x[i - 1] + std_x[end]*randn()
        end
    end
    return x, std_x, z
end

reals, std_x, upper_rw = generate_swtiching_hgf(n_samples, switches)
obs = reals .+ sqrt(0.01)*randn(length(reals))
dims = 2

pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,3,'0')) # Left-pads a number with zeros, converts it to symbol and appends to sym

function generate_algorithm(ndim, n_samples)
    fg = FactorGraph()
    z = Vector{Variable}(undef, n_samples)
    x = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)
    s = Vector{Variable}(undef, n_samples)
    @RV A ~ Dirichlet(ones(2, 2))
    @RV [id=pad(:z,1)] z[1] ~ GaussianMeanPrecision(placeholder(:mz_prior1), placeholder(:wz_prior1))
    @RV [id=pad(:x,1)] x[1] ~ GaussianMeanPrecision(placeholder(:mx_prior1), placeholder(:wx_prior1))
    @RV [id=pad(:y,1)] y[1] ~ GaussianMeanPrecision(x[1], placeholder(:wy_prior1))
    @RV [id=pad(:s,1)] s[1] ~ ForneyLab.Categorical(ones(ndim) ./ ndim)
    placeholder(y[1], :y, index = 1)
    for t in 2:n_samples
        @RV [id=pad(:s,t)] s[t] ~ Transition(s[t-1], A)
        @RV [id=pad(:z,t)] z[t] ~ GaussianMeanPrecision(z[t - 1], placeholder(pad(:wz_transition, t)))
        @RV [id=pad(:x,t)] x[t] ~ SwitchingGaussianControlledVariance(x[t - 1], z[t],[1.0, 1.0],[-2.0, 2.0],s[t])
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end
    q = PosteriorFactorization(x, z ,s, A, ids=[:X :Z :S :A])
    algo = messagePassingAlgorithm(free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

include("compatibility.jl")
code = generate_algorithm(2, n_samples)
eval(Meta.parse(code))

function infer(obs;
    ndims = 2,
    wy_prior1 = 1.0,
    κ_m_prior = [1.0, 1.0],
    ω_m_prior = [-2.0, 2.0],
    κ_w_prior =  huge .* diageye(ndims),
    ω_w_prior = huge .* diageye(ndims),
    z_m_prior = 0.0,
    z_w_prior = 100.0,
    x_m_prior = 0.0,
    x_w_prior = 1.0,
    x_x_m_prior = zeros(ndims),
    x_x_w_prior = 10.0*diageye(ndims),
    z_z_m_prior = zeros(ndims),
    z_z_w_prior = 100.0*diageye(ndims),
    z_w_transition_prior = 100.0,
    y_w_transition_prior =  1/0.01,
)

    marginals = Dict()
    marginals[:A] = ProbabilityDistribution(ForneyLab.MatrixVariate, Dirichlet, a=ones(2, 2))
    marginals[:κ] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = κ_m_prior, w = κ_w_prior)
    marginals[:ω] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = ω_m_prior, w = ω_w_prior)
    marginals[pad(:z,1)] = vague(GaussianMeanPrecision)
    marginals[pad(:x,1)] = vague(GaussianMeanPrecision)
    marginals[pad(:s,1)] = vague(Categorical, 2)
    for t = 2:n_samples
        marginals[pad(:z,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = z_m_prior, w = z_w_prior)
        marginals[pad(:x,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = x_m_prior, w = x_w_prior)
        marginals[pad(:s,t)] = ProbabilityDistribution(Categorical, p = [0.5, 0.5])
        marginals[pad(:s,t)*:_*pad(:s,t-1)] = ProbabilityDistribution(Contingency,p=[0.5 0.5; 0.5 0.5])
        marginals[pad(:z,t)*:_*pad(:z,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = z_z_m_prior, w = z_z_w_prior)
        marginals[pad(:x,t)*:_*pad(:x,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = x_x_m_prior, w = x_x_w_prior)
    end
    data = Dict()
    data[:y] = obs
    data[:mz_prior1] = z_m_prior
    data[:wz_prior1] = z_w_prior
    data[:mx_prior1] = x_m_prior
    data[:wx_prior1] = x_w_prior
    data[:wy_prior1] = wy_prior1
    for t = 1:n_samples
        data[pad(:wz_transition, t)] = z_w_transition_prior
        data[pad(:wy_transition, t)] = y_w_transition_prior
    end


    n_its = 20
    fe = Vector{Float64}(undef, n_its)
    ##
    @showprogress "Iterations" for i = 1:n_its
        stepX!(data, marginals)
        stepZ!(data, marginals)
        stepA!(data, marginals)
        stepS!(data, marginals)
        # stepO!(data, marginals)
        # stepK!(data, marginals)
        fe[i] = freeEnergy(data, marginals)
    end

    mz = [ForneyLab.unsafeMean(marginals[pad(:z,t)]) for t=1:n_samples]
    vz = [ForneyLab.unsafeVar(marginals[pad(:z,t)]) for t=1:n_samples]
    mx = [ForneyLab.unsafeMean(marginals[pad(:x,t)]) for t=1:n_samples]
    vx = [ForneyLab.unsafeVar(marginals[pad(:x,t)]) for t=1:n_samples]
    ms = [ForneyLab.unsafeMean(marginals[pad(:s,t)]) for t=1:n_samples]
    return mz,vz,mx,vx,ms,fe
end

mz,vz,mx,vx,ms,fe = infer(obs)

plot(mx, ribbon=sqrt.(vx))
plot!(reals)
scatter!(obs)

plot(mz, ribbon=sqrt.(vz))
plot!(upper_rw)

categories = [x[2] for x in findmax.(ms)]
scatter(categories)
scatter!(switches)

plot(fe)
