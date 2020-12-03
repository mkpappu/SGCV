# using Revise
using ForneyLab
include("sgcv/SGCV.jl")
using Main.SGCV
using GCV
using Plots
using SparseArrays
using Random
using ProgressMeter
include("compatibility.jl")


pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,3,'0')) # Left-pads a number with zeros, converts it to symbol and appends to sym

function generate_mp(n_cats1, n_cats2, n_samples)
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

function mp(obs;
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
    z1_w_transition_prior = 10.0,
    z2_z2_m_prior = zeros(n_cats2),
    z2_z2_w_prior = 1.0*diageye(n_cats2),
    z2_w_transition_prior = 1000.0,
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

include("generator3l.jl")
index=3
obs = dataset[index]["obs"]
mnv = dataset[index]["nv"]
omegas1 = dataset[index]["ωs1"]
omegas2 = dataset[index]["ωs2"]
switches1 = dataset[index]["switches1"]
switches2 = dataset[index]["switches2"]
code = generate_mp(n_cats1, n_cats2, n_samples)
eval(Meta.parse(code))
mz1, vz1, mω1, vω1, mz2, vz2, mω2, vω2, mx, vx,ms1, ms2, fe = mp(obs, n_cats1=3, n_cats2=2,
                  ω1_m_prior=omegas1 .+ sqrt(1)*randn(length(omegas1)),
                  ω2_m_prior=omegas2 .+ sqrt(1)*randn(length(omegas2)),
                  y_w_transition_prior=1/mnv)


plot(mx, ribbon=sqrt.(vx))
first_layer = dataset[index]["reals"]
plot!(first_layer)

plot(mz2, ribbon=sqrt.(vz2))
rw2 = dataset[index]["rw2"]
plot!(rw2)


plot(mz1, ribbon=sqrt.(vz1))
rw1 = dataset[index]["rw1"]
plot!(rw1)

categories1 = [x[2] for x in findmax.(ms1)]
scatter(categories1)
scatter!(switches1)

categories2 = [x[2] for x in findmax.(ms2)]
scatter(categories2)
scatter!(switches2)

println(mω1," ", vω1)
println(mω2," ", vω2)
plot(fe[2:end])
