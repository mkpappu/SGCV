using Revise
using ForneyLab
using ProgressMeter
using GCV
using Plots
using SparseArrays
using Random
include("compatibility.jl")

Random.seed!(10)

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

ω1, ω2 = -2.0, 2.0
f(s) = s[1]*ω1 + s[2]*ω2

function generate_sampler(dims, n_samples)

    fg = FactorGraph()

    s = Vector{Variable}(undef, n_samples)
    ω = Vector{Variable}(undef, n_samples)
    x = Vector{Variable}(undef, n_samples)
    z = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)


    ω1, ω2 = -2.0, 2.0
    f(s) = s[1]*ω1 + s[2]*ω2
    @RV [id=pad(:z,1)] z[1] ~ GaussianMeanPrecision(placeholder(:mz_prior1), placeholder(:wz_prior1))
    @RV [id=pad(:s,1)] s[1] ~ Categorical(ones(dims)/dims)
    @RV A ~ Dirichlet(ones(dims, dims))
    @RV [id=pad(:x,1)] x[1] ~ GaussianMeanPrecision(placeholder(:mx_prior1), placeholder(:wx_prior1))
    for t in 2:n_samples
        global s_t_min, x_t_min, z_t_min
        @RV [id=pad(:s,t)] s[t] ~ Transition(s[t-1], A)
        @RV [id=pad(:ω,t)] ω[t] ~ Nonlinear{Sampling}(s[t],g=f)
        @RV [id=pad(:z,t)] z[t] ~ GaussianMeanPrecision(z[t-1], placeholder(pad(:wz_transition, t)))
        @RV [id=pad(:x,t)] x[t] ~ GaussianControlledVariance(x[t-1], z[t], 1.0, ω[t])
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end

    pfz = PosteriorFactorization()

    q_A = PosteriorFactor(A, id=:A)

    q_s = Vector{PosteriorFactor}(undef, n_samples)
    q_x = Vector{PosteriorFactor}(undef, n_samples)
    for t in 1:n_samples
        q_s[t] = PosteriorFactor(s[t],id=:S_*t)
    end
    q_z = PosteriorFactor(z,id=:Z)
    q_x = PosteriorFactor(x,id=:X)
    # Compile algorithm
    algo_mf = messagePassingAlgorithm(id=:MF, free_energy=true)

    src_code = algorithmSourceCode(algo_mf, free_energy=true)

    return src_code
end

code = generate_sampler(2, n_samples)

eval(Meta.parse(code))

function mp_sampler(obs;
    dims = 2,
    κ_m_prior = [1.0, 1.0],
    ω_m_prior = [-2.0, 2.0],
    z_m_prior = 0.0,
    z_w_prior = 100.0,
    x_m_prior = 0.0,
    x_w_prior = 1.0,
    z_w_transition_prior = 100.0,
    y_w_transition_prior =  1/0.01,
)

    # Initial posterior factors
    marginals = Dict{Symbol, ProbabilityDistribution}(:A => vague(Dirichlet, (2,2)))
    for t in 1:n_samples
        marginals[pad(:s,t)] = ProbabilityDistribution(Univariate, Categorical, p=0.5*ones(dims))
        marginals[pad(:ω,t)] = vague(SampleList)
        marginals[pad(:x,t)] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=x_w_prior, w=x_w_prior)
        marginals[pad(:z,t)] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=z_m_prior, w=z_w_prior)

    end
    data = Dict()
    data[:y] = obs
    data[:mz_prior1] = z_m_prior
    data[:wz_prior1] = z_w_prior
    data[:mx_prior1] = x_m_prior
    data[:wx_prior1] = x_w_prior
    for t = 1:n_samples
        data[pad(:wz_transition, t)] = z_w_transition_prior
        data[pad(:wy_transition, t)] = y_w_transition_prior
    end


    # Run algorithm
    fe = []
    n_its = 100
    @showprogress for i in 1:n_its
        stepMFA!(data, marginals)
        stepMFX!(data, marginals)
        stepMFZ!(data, marginals)
        for k in 1:n_samples
            step!(:MFS_*k, data, marginals)
        end
        push!(fe, freeEnergyMF(data, marginals))
    end
    ;

    mz = [ForneyLab.unsafeMean(marginals[pad(:z,t)]) for t=1:n_samples]
    vz = [ForneyLab.unsafeVar(marginals[pad(:z,t)]) for t=1:n_samples]
    mx = [ForneyLab.unsafeMean(marginals[pad(:x,t)]) for t=1:n_samples]
    vx = [ForneyLab.unsafeVar(marginals[pad(:x,t)]) for t=1:n_samples]
    ms = [ForneyLab.unsafeMean(marginals[pad(:s,t)]) for t=1:n_samples]
    return mz,vz,mx,vx,ms,fe
end

mz,vz,mx,vx,ms,fe = mp_sampler(obs)

plot(mz, ribbon=sqrt.((vz)))
plot!(upper_rw)
plot!(std_x)

m_switches = [x[2] for x in findmax.(ms)]
scatter(m_switches)
scatter!(switches)

plot(mx, ribbon=sqrt.(vx))
plot!(reals)
scatter!(obs)

plot(fe)
