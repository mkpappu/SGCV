using Revise
using ForneyLab
include("sgcv/SGCV.jl")
using Main.SGCV
include("gcv/GCV.jl")
using Main.GCV
using Plots
using SparseArrays
using Random
using ProgressMeter
include("compatibility.jl")

pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,3,'0')) # Left-pads a number with zeros, converts it to symbol and appends to sym

# 2 layers
function generate_mp_2l(ndim, n_samples)
    fg = FactorGraph()
    z = Vector{Variable}(undef, n_samples)
    x = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)
    s = Vector{Variable}(undef, n_samples)
    @RV A ~ Dirichlet(ones(ndim, ndim))
    @RV ω ~ GaussianMeanPrecision(placeholder(:mω, dims=(ndim, )), placeholder(:wω, dims=(ndim, ndim)))
    @RV [id=pad(:z,1)] z[1] ~ GaussianMeanPrecision(placeholder(:mz_prior1), placeholder(:wz_prior1))
    @RV [id=pad(:x,1)] x[1] ~ GaussianMeanPrecision(placeholder(:mx_prior1), placeholder(:wx_prior1))
    @RV [id=pad(:y,1)] y[1] ~ GaussianMeanPrecision(x[1], placeholder(:wy_prior1))
    @RV [id=pad(:s,1)] s[1] ~ ForneyLab.Categorical(ones(ndim) ./ ndim)
    placeholder(y[1], :y, index = 1)
    for t in 2:n_samples
        @RV [id=pad(:s,t)] s[t] ~ Transition(s[t-1], A)
        @RV [id=pad(:z,t)] z[t] ~ GaussianMeanPrecision(z[t - 1], placeholder(pad(:wz_transition, t)))
        @RV [id=pad(:x,t)] x[t] ~ SwitchingGaussianControlledVariance(x[t - 1], z[t],ones(ndim), ω,s[t])
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end
    q = PosteriorFactorization(x, z, ω, s, A, ids=[:X :Z :Ω :S :A])
    algo = messagePassingAlgorithm(free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

function mp_2l(obs;
    ndims,
    n_its = 100,
    wy_prior1 = 1.0,
    κ_m_prior = ones(ndims),
    ω_m_prior = omegas,
    κ_w_prior =  huge .* diageye(ndims),
    ω_w_prior = 1.0 * diageye(ndims),
    z_m_prior = 0.0,
    z_w_prior = 1.0,
    x_m_prior = 0.0,
    x_w_prior = 1.0,
    x_x_m_prior = zeros(ndims),
    x_x_w_prior = 1.0*diageye(ndims),
    z_z_m_prior = zeros(ndims),
    z_z_w_prior = 1.0*diageye(ndims),
    z_w_transition_prior = 100.0,
    y_w_transition_prior =  1/mnv,
)
    n_samples = length(obs)
    marginals = Dict()
    marginals[:A] = ProbabilityDistribution(ForneyLab.MatrixVariate, Dirichlet, a=ones(ndims, ndims))
    marginals[:κ] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = κ_m_prior, w = κ_w_prior)
    marginals[:ω] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = ω_m_prior, w = ω_w_prior)
    marginals[pad(:z,1)] = vague(GaussianMeanPrecision)
    marginals[pad(:x,1)] = vague(GaussianMeanPrecision)
    marginals[pad(:s,1)] = vague(Categorical, ndims)
    for t = 2:n_samples
        marginals[pad(:z,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = z_m_prior, w = z_w_prior)
        marginals[pad(:x,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = x_m_prior, w = x_w_prior)
        marginals[pad(:s,t)] = ProbabilityDistribution(Categorical, p = ones(ndims) ./ ndims)
        marginals[pad(:s,t)*:_*pad(:s,t-1)] = ProbabilityDistribution(Contingency,p=ones(ndims, ndims) ./ ndims)
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
    data[:mω] = ω_m_prior
    data[:wω] = ω_w_prior
    for t = 1:n_samples
        data[pad(:wz_transition, t)] = z_w_transition_prior
        data[pad(:wy_transition, t)] = y_w_transition_prior
    end


    fe = Vector{Float64}(undef, n_its)

    @showprogress "Iterations" for i = 1:n_its

        stepX!(data, marginals)
        stepS!(data, marginals)
        stepA!(data, marginals)
        stepΩ!(data, marginals)
        stepZ!(data, marginals)

        fe[i] = freeEnergy(data, marginals)
    end

    mz = [ForneyLab.unsafeMean(marginals[pad(:z,t)]) for t=1:n_samples]
    vz = [ForneyLab.unsafeVar(marginals[pad(:z,t)]) for t=1:n_samples]
    mω = ForneyLab.unsafeMean(marginals[:ω])
    vω = ForneyLab.unsafeCov(marginals[:ω])
    mx = [ForneyLab.unsafeMean(marginals[pad(:x,t)]) for t=1:n_samples]
    vx = [ForneyLab.unsafeVar(marginals[pad(:x,t)]) for t=1:n_samples]
    ms = [ForneyLab.unsafeMean(marginals[pad(:s,t)]) for t=1:n_samples]
    return mz,vz,mω, vω, mx,vx,ms,fe
end

# 3 layers
function generate_mp_3l(n_cats1, n_cats2, n_samples)
    fg = FactorGraph()
    z2 = Vector{Variable}(undef, n_samples)
    s2 = Vector{Variable}(undef, n_samples)
    z1 = Vector{Variable}(undef, n_samples)
    s1 = Vector{Variable}(undef, n_samples)
    x = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)
    @RV A2 ~ Dirichlet(ones(n_cats2, n_cats2))
    @RV A1 ~ Dirichlet(ones(n_cats1, n_cats1))
    @RV ω2 ~ GaussianMeanPrecision(placeholder(:mω2, dims=(n_cats2, )), placeholder(:wω2, dims=(n_cats2, n_cats2)))
    @RV ω1 ~ GaussianMeanPrecision(placeholder(:mω1, dims=(n_cats1, )), placeholder(:wω1, dims=(n_cats1, n_cats1)))
    @RV [id=pad(:z1,1)] z1[1] ~ GaussianMeanPrecision(placeholder(:mz1_prior1), placeholder(:wz1_prior1))
    @RV [id=pad(:z2,1)] z2[1] ~ GaussianMeanPrecision(placeholder(:mz2_prior1), placeholder(:wz2_prior1))
    @RV [id=pad(:x,1)] x[1] ~ GaussianMeanPrecision(placeholder(:mx_prior1), placeholder(:wx_prior1))
    @RV [id=pad(:y,1)] y[1] ~ GaussianMeanPrecision(x[1], placeholder(:wy_prior1))
    @RV [id=pad(:s1,1)] s1[1] ~ ForneyLab.Categorical(ones(n_cats1) ./ n_cats1)
    @RV [id=pad(:s2,1)] s2[1] ~ ForneyLab.Categorical(ones(n_cats2) ./ n_cats2)
    placeholder(y[1], :y, index = 1)
    for t in 2:n_samples
        @RV [id=pad(:s2, t)] s2[t] ~ Transition(s2[t-1], A2)
        @RV [id=pad(:s1, t)] s1[t] ~ Transition(s1[t-1], A1)
        @RV [id=pad(:z2,t)] z2[t] ~ GaussianMeanPrecision(z2[t - 1], placeholder(pad(:wz2_transition, t)))
        @RV [id=pad(:z1,t)] z1[t] ~ SwitchingGaussianControlledVariance(z1[t - 1], z2[t], ones(n_cats2), ω2,s2[t])
        @RV [id=pad(:x,t)] x[t] ~ SwitchingGaussianControlledVariance(x[t - 1], z1[t], ones(n_cats1), ω1,s1[t])
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end
    q = PosteriorFactorization(x, z1, s1, z2, s2, A1, A2, ω1, ω2, ids=[:X :Z1 :S1 :Z2 :S2 :A1 :A2 :Ω1 :Ω2])
    algo = messagePassingAlgorithm(free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

function mp_3l(obs;
    n_cats1, n_cats2,
    n_its = 20,
    wy_prior1 = 1.0,
    κ1_m_prior = ones(n_cats1),
    ω1_m_prior = omegas1,
    κ1_w_prior =  huge .* diageye(n_cats1),
    ω1_w_prior = 1.0 * diageye(n_cats1),
    z1_m_prior = 0.0,
    z1_w_prior = 1.0,
    κ2_m_prior = ones(n_cats2),
    ω2_m_prior = omegas2,
    κ2_w_prior =  huge .* diageye(n_cats2),
    ω2_w_prior = 1.0 * diageye(n_cats2),
    z2_m_prior = 0.0,
    z2_w_prior = 1.0,
    x_m_prior = 0.0,
    x_w_prior = 1.0,
    x_x_m_prior = zeros(n_cats1),
    x_x_w_prior = 1.0*diageye(n_cats1),
    z1_z1_m_prior = zeros(n_cats1),
    z1_z1_w_prior = 1.0*diageye(n_cats1),
    z1_w_transition_prior = 1000.0,
    z2_z2_m_prior = zeros(n_cats2),
    z2_z2_w_prior = 1.0*diageye(n_cats2),
    z2_w_transition_prior = 10.0,
    y_w_transition_prior =  1/mnv,
)
    n_samples = length(obs)
    marginals = Dict()

    # second  layer
    marginals[:A1] = ProbabilityDistribution(ForneyLab.MatrixVariate, Dirichlet, a=ones(n_cats1, n_cats1))
    marginals[:κ1] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = κ1_m_prior, w = κ1_w_prior)
    marginals[:ω1] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = ω1_m_prior, w = ω1_w_prior)
    marginals[pad(:z1,1)] = vague(GaussianMeanPrecision)

    # third layer
    marginals[:A2] = ProbabilityDistribution(ForneyLab.MatrixVariate, Dirichlet, a=ones(n_cats2, n_cats2))
    marginals[:κ2] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = κ2_m_prior, w = κ2_w_prior)
    marginals[:ω2] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m = ω2_m_prior, w = ω2_w_prior)
    marginals[pad(:z2,1)] = vague(GaussianMeanPrecision)

    marginals[pad(:x,1)] = vague(GaussianMeanPrecision)
    marginals[pad(:s1,1)] = vague(Categorical, n_cats1)
    marginals[pad(:s2,1)] = vague(Categorical, n_cats2)
    for t = 2:n_samples
        marginals[pad(:z1,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = z1_m_prior, w = z1_w_prior)
        marginals[pad(:z2,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = z2_m_prior, w = z2_w_prior)
        marginals[pad(:x,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = x_m_prior, w = x_w_prior)
        marginals[pad(:s1,t)] = ProbabilityDistribution(Categorical, p = ones(n_cats1) ./ n_cats1)
        marginals[pad(:s2,t)] = ProbabilityDistribution(Categorical, p = ones(n_cats2) ./ n_cats2)
        marginals[pad(:s1,t)*:_*pad(:s1,t-1)] = ProbabilityDistribution(Contingency,p=ones(n_cats1, n_cats1) ./ n_cats1)
        marginals[pad(:s2,t)*:_*pad(:s2,t-1)] = ProbabilityDistribution(Contingency,p=ones(n_cats2, n_cats2) ./ n_cats2)
        marginals[pad(:z1,t)*:_*pad(:z1,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = z1_z1_m_prior, w = z2_z2_m_prior)
        marginals[pad(:z2,t)*:_*pad(:z2,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = z2_z2_m_prior, w = z2_z2_w_prior)
        marginals[pad(:x,t)*:_*pad(:x,t-1)] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = x_x_m_prior, w = x_x_w_prior)
    end
    data = Dict()
    data[:y] = obs
    data[:mz1_prior1] = z1_m_prior
    data[:wz1_prior1] = z1_w_prior
    data[:mz2_prior1] = z2_m_prior
    data[:wz2_prior1] = z2_w_prior
    data[:mx_prior1] = x_m_prior
    data[:wx_prior1] = x_w_prior
    data[:wy_prior1] = wy_prior1
    data[:mω1] = ω1_m_prior
    data[:mω2] = ω2_m_prior
    data[:wω1] = ω1_w_prior
    data[:wω2] = ω2_w_prior
    for t = 1:n_samples
        data[pad(:wz1_transition, t)] = z1_w_transition_prior
        data[pad(:wz2_transition, t)] = z2_w_transition_prior
        data[pad(:wy_transition, t)] = y_w_transition_prior
    end


    fe = Vector{Float64}(undef, n_its)

    @showprogress "Iterations" for i = 1:n_its

        stepX!(data, marginals)
        stepS1!(data, marginals)
        stepA1!(data, marginals)
        stepΩ1!(data, marginals)
        stepZ1!(data, marginals)
        stepS2!(data, marginals)
        stepA2!(data, marginals)
        stepΩ2!(data, marginals)
        stepZ2!(data, marginals)

        fe[i] = freeEnergy(data, marginals)
    end

    mz1 = [ForneyLab.unsafeMean(marginals[pad(:z1,t)]) for t=1:n_samples]
    vz1 = [ForneyLab.unsafeVar(marginals[pad(:z1,t)]) for t=1:n_samples]
    mω1 = ForneyLab.unsafeMean(marginals[:ω1])
    vω1 = ForneyLab.unsafeCov(marginals[:ω1])
    mz2 = [ForneyLab.unsafeMean(marginals[pad(:z2,t)]) for t=1:n_samples]
    vz2 = [ForneyLab.unsafeVar(marginals[pad(:z2,t)]) for t=1:n_samples]
    mω2 = ForneyLab.unsafeMean(marginals[:ω2])
    vω2 = ForneyLab.unsafeCov(marginals[:ω2])
    mx = [ForneyLab.unsafeMean(marginals[pad(:x,t)]) for t=1:n_samples]
    vx = [ForneyLab.unsafeVar(marginals[pad(:x,t)]) for t=1:n_samples]
    ms1 = [ForneyLab.unsafeMean(marginals[pad(:s1,t)]) for t=1:n_samples]
    ms2 = [ForneyLab.unsafeMean(marginals[pad(:s2,t)]) for t=1:n_samples]
    return mz1, vz1, mω1, vω1, mz2, vz2, mω2, vω2, mx, vx, ms1, ms2, fe
end

# 1 layer
function generate_mp(n_samples)
    fg = FactorGraph()
    z = Vector{Variable}(undef, n_samples)
    x = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)

    @RV ω ~ GaussianMeanPrecision(placeholder(:mω), placeholder(:wω))
    @RV [id=pad(:z,1)] z[1] ~ GaussianMeanPrecision(placeholder(:mz_prior1), placeholder(:wz_prior1))
    @RV [id=pad(:x,1)] x[1] ~ GaussianMeanPrecision(placeholder(:mx_prior1), placeholder(:wx_prior1))
    @RV [id=pad(:y,1)] y[1] ~ GaussianMeanPrecision(x[1], placeholder(:wy_prior1))
    placeholder(y[1], :y, index = 1)
    for t in 2:n_samples
        @RV [id=pad(:z,t)] z[t] ~ GaussianMeanPrecision(z[t - 1], placeholder(pad(:wz_transition, t)))
        @RV [id=pad(:x,t)] x[t] ~ GaussianControlledVariance(x[t - 1], z[t], 1.0, ω)
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end
    q = PosteriorFactorization(x, z, ω, ids=[:X :Z :Ω])
    algo = messagePassingAlgorithm(free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

function mp(obs;
    n_its = 100,
    wy_prior1 = 1.0,
    ω_m_prior = omega,
    ω_w_prior = 1.0,
    z_m_prior = 0.0,
    z_w_prior = 100.0,
    x_m_prior = 0.0,
    x_w_prior = 1.0,
    x_x_m_prior = zeros(2),
    x_x_w_prior = 1.0*diageye(2),
    z_z_m_prior = zeros(2),
    z_z_w_prior = 100.0*diageye(2),
    z_w_transition_prior = 1000.0,
    y_w_transition_prior =  1/mnv,
)
    n_samples = length(obs)
    marginals = Dict()
    marginals[:ω] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = ω_m_prior, w = ω_w_prior)
    marginals[pad(:z,1)] = vague(GaussianMeanPrecision)
    marginals[pad(:x,1)] = vague(GaussianMeanPrecision)
    for t = 2:n_samples
        marginals[pad(:z,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = z_m_prior, w = z_w_prior)
        marginals[pad(:x,t)] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = x_m_prior, w = x_w_prior)
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
    data[:mω] = ω_m_prior
    data[:wω] = ω_w_prior
    for t = 1:n_samples
        data[pad(:wz_transition, t)] = z_w_transition_prior
        data[pad(:wy_transition, t)] = y_w_transition_prior
    end


    fe = Vector{Float64}(undef, n_its)

    @showprogress "Iterations" for i = 1:n_its

        stepX!(data, marginals)
        stepZ!(data, marginals)
        stepΩ!(data, marginals)

        fe[i] = freeEnergy(data, marginals)
    end

    mz = [ForneyLab.unsafeMean(marginals[pad(:z,t)]) for t=1:n_samples]
    vz = [ForneyLab.unsafeVar(marginals[pad(:z,t)]) for t=1:n_samples]
    mω = ForneyLab.unsafeMean(marginals[:ω])
    vω = ForneyLab.unsafeCov(marginals[:ω])
    mx = [ForneyLab.unsafeMean(marginals[pad(:x,t)]) for t=1:n_samples]
    vx = [ForneyLab.unsafeVar(marginals[pad(:x,t)]) for t=1:n_samples]
    return mz,vz,mω, vω, mx,vx,fe
end

using JLD
results2shgf = load("dump/results_2shgf_stocks_mixture.jld")["results"]
resultshgf = load("dump/results_hgf_stocks_mixture.jld")["results"]
results3shgf = load("dump/results_3shgf_stocks_mixture.jld")["results"]
plot(results2shgf["fe"], label="2-L SHGF")
plot!(results3shgf["fe"], label="3-L SHGF")
plot!(resultshgf["fe"], label="HGF")
savefig("figures/fe_stocks.pdf")

# download data
using CSV
using DataFrames
using Plots
df = CSV.File("data/AAPL.csv") |> DataFrame
plot(df[:Open])
series = df[!, :Open]


# good prior
omegas = [-1.0, 4.0]
n_cats = length(omegas)
code = generate_mp_2l(n_cats, length(series))
eval(Meta.parse(code))
mz,vz,mω, vω, mx,vx,ms,fe = mp_2l(series, x_m_prior=series[1], ndims=n_cats, z_w_prior=100.0, z_z_w_prior=100.0*diageye(2), z_w_transition_prior=100.0, ω_m_prior=omegas, ω_w_prior=diageye(2), y_w_transition_prior=1.0)

plot(mx, ribbon=sqrt.(vx))
scatter!(series)
plot(mz, ribbon=sqrt.(vz))
categories = [x[2] for x in findmax.(ms)]
scatter(categories)
plot(fe)

results =   Dict("mz" => mz, "vz" => vz,
                  "mx" => mx, "vx" => vx,
                  "ms" => ms, "fe" => fe,
                  "mω" => mω, "vω" => vω,
                  "ωprior" => omegas)

using JLD

JLD.save("dump/results_2shgf_stocks_mixture.jld","results",results)

omegas1 = [-1.0, 4.0]
omegas2 = [-3.0, 1.0]
code = generate_mp_3l(2, 2, length(series))
eval(Meta.parse(code))


mz1, vz1, mω1, vω1, mz2, vz2, mω2, vω2, mx1, vx1, ms1, ms2, fe2 = mp_3l(series, n_its = 100, n_cats1=2, n_cats2=2,
                                                                        x_m_prior=series[1],
                                                                        ω1_m_prior=omegas1, ω2_m_prior=omegas2,
                                                                        ω1_w_prior = 1.0 * diageye(2),
                                                                        ω2_w_prior = 10.0 * diageye(2),
                                                                        z1_w_transition_prior = 100.0,
                                                                        z1_w_prior = 100.0,
                                                                        z2_w_prior = 100.0,
                                                                        y_w_transition_prior=1.0)
plot(mx1, ribbon=sqrt.(vx1))
scatter!(series)
plot(mz1, ribbon=sqrt.(vz1))
categories1 = [x[2] for x in findmax.(ms1)]
scatter(categories1)
plot(mz2, ribbon=sqrt.(vz2))
categories2 = [x[2] for x in findmax.(ms2)]
scatter(categories2)
plot(fe2)

results =   Dict("mz1" => mz1, "vz1" => vz1,
                 "mz2" => mz2, "vz2" => vz2,
                 "mx1" => mx1, "vx" => vx1,
                 "ms1" => ms1, "ms2" => ms2,
                 "fe" => fe2,
                 "mω1" => mω1, "vω1" => vω1,
                 "mω2" => mω2, "vω2" => vω2,
                 "ωprior1" => omegas1,
                 "ωprior2" => omegas2)

using JLD
JLD.save("dump/results_3shgf_stocks_mixture.jld","results",results)

# NOTE: For running normal GCV you'd have to kill Julia at first
# compile pad, generate_mp and mp functions
using ForneyLab
using GCV
using CSV
using DataFrames
using Plots
using ProgressMeter
include("compatibility.jl")

df = CSV.File("data/AAPL.csv") |> DataFrame
plot(df[:Open])
series = df[!, :Open]

omega  = 1.0
kappa = 1.0
src_code = generate_mp(length(series))
eval(Meta.parse(src_code));
mz0,vz0,mω0, vω0, mx0,vx0,fe0 = mp(series, x_m_prior=series[1], z_z_w_prior=100.0 * diageye(2), z_w_transition_prior=100.0, ω_m_prior=omega, ω_w_prior=1.0, y_w_transition_prior=1.0);

plot(mx0, ribbon=sqrt.(vx0))
scatter!(series)
plot(mz0, ribbon=sqrt.(vz0))
plot(fe0)

results =   Dict("mz" => mz0, "vz" => vz0,
                  "mx" => mx0, "vx" => vx0,
                  "fe" => fe0,
                  "mω" => mω0, "vω" => vω0,
                  "ωprior" => omega)

using JLD
JLD.save("dump/results_hgf_stocks_mixture.jld","results",results)
