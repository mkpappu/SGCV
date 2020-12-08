using Revise
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

function generate_mp(ndim, n_samples)
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
        @RV [id=pad(:x,t)] x[t] ~ SwitchingGaussianControlledVariance(x[t - 1], z[t],ones(ndim),ω,s[t])
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:wy_transition, t)))
        placeholder(y[t], :y, index = t)
    end
    q = PosteriorFactorization(x, z ,s, A, ω, ids=[:X :Z :S :A :Ω])
    algo = messagePassingAlgorithm(free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

function mp(obs;
    ndims,
    n_its = 100,
    wy_prior1 = 1.0,
    κ_m_prior = ones(ndims),
    ω_m_prior = omegas,
    κ_w_prior =  huge .* diageye(ndims),
    ω_w_prior = 1.0 .* diageye(ndims),
    z_m_prior = 0.0,
    z_w_prior = 10.0,
    x_m_prior = 0.0,
    x_w_prior = 1.0,
    x_x_m_prior = zeros(ndims),
    x_x_w_prior = 1.0*diageye(ndims),
    z_z_m_prior = zeros(ndims),
    z_z_w_prior = 10.0*diageye(ndims),
    z_w_transition_prior = 1000.0,
    y_w_transition_prior =  1/mnv,
)
    println("JULIA IS FUCKING SHIT! I HATE YOU!!!")

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
    println("fucking SHIT!")
    for i = 1:n_its

        stepX!(data, marginals)
        stepS!(data, marginals)
        stepA!(data, marginals)
        stepZ!(data, marginals)
        stepΩ!(data, marginals)

        fe[i] = freeEnergy(data, marginals)
    end

    mz = [ForneyLab.unsafeMean(marginals[pad(:z,t)]) for t=1:n_samples]
    vz = [ForneyLab.unsafeVar(marginals[pad(:z,t)]) for t=1:n_samples]
    mx = [ForneyLab.unsafeMean(marginals[pad(:x,t)]) for t=1:n_samples]
    vx = [ForneyLab.unsafeVar(marginals[pad(:x,t)]) for t=1:n_samples]
    ms = [ForneyLab.unsafeMean(marginals[pad(:s,t)]) for t=1:n_samples]
    mω = ForneyLab.unsafeMean(marginals[:ω])
    vω = ForneyLab.unsafeCov(marginals[:ω])
    return mz,vz,mx,vx,ms, mω, vω, fe
end

include("generator.jl")

code = generate_mp(n_cats, n_samples)
eval(Meta.parse(code))
results = Dict()

@showprogress "Datasets" for i in 1:n_datasets
    obs = dataset[i]["obs"]
    mnv = dataset[i]["nv"]
    omegas = dataset[i]["ωs"] .+ sqrt(1.0)*randn(length(dataset[i]["ωs"]))
    try
        mz,vz,mx,vx,ms, mω, vω,fe = mp(obs, ndims=n_cats, ω_m_prior=omegas,
                                      ω_w_prior=diageye(n_cats),
                                      y_w_transition_prior=1/mnv)
        results[i] = Dict("mz" => mz, "vz" => vz,
                          "mx" => mx, "vx" => vx,
                          "ms" => ms, "fe" => fe,
                          "mω" => mω, "vω" => vω,
                          "ωprior" => omegas)
    catch e
           println("Failed $(i)")
    end
end

index = 15

mz, vz, mx, vx, ms, mω, vω,fe = results[index]["mz"], results[index]["vz"], results[index]["mx"], results[index]["vx"], results[index]["ms"], results[index]["mω"], results[index]["vω"], results[index]["fe"]
reals = dataset[index]["reals"]
obs = dataset[index]["obs"]
upper_rw = dataset[index]["rw"]
switches = dataset[index]["switches"]
omegas = dataset[index]["ωs"]


# Plot recovered data
categories = [x[2] for x in findmax.(ms)]
maxup = maximum(obs) + 1.0
mindown = minimum(obs) - 1.0
plot()
for (index, categ) in enumerate(categories)
    if categ == 1
        scatter!([index], [maxup], color=:green, markershape=:xcross, markersize=2, markeralpha=0.4, label="")
    elseif categ == 2
        scatter!([index], [maxup], color=:blue, markershape=:xcross, markersize=2, markeralpha=0.4, label="")
    else
        scatter!([index], [maxup],  color=:red, markershape=:xcross, markersize=2, markeralpha=0.4, label="")
    end

end
for (index, categ) in enumerate(switches)
    if categ == 1
        scatter!([index], [mindown], color=:green, markershape=:xcross, markersize=2, markeralpha=0.4, label="")
    elseif categ == 2
        scatter!([index], [mindown], color=:blue, markershape=:xcross, markersize=2, markeralpha=0.4, label="")
    else
        scatter!([index], [mindown],  color=:red, markershape=:xcross, markersize=2, markeralpha=0.4, label="")
    end

end
plot!(mx, ribbon=sqrt.(vx), label="inferred")
plot!(reals, label="real")
scatter!(obs, color=:grey, markershape=:xcross, markersize=2, markeralpha=0.4, label="observed")
savefig("figures/recovered_switches.pdf")

using LaTeXStrings
plot(mz, ribbon=sqrt.(vz), label="inferred", ylabel=L"x^{(1)}", xlabel=L"t")
plot!(upper_rw, label="real")
savefig("figures/upper_layer.pdf")


plot(fe[2:end])

sum = 0
for i in 1:n_datasets
    try
        mω = results[i]["mω"]
        mse_ω = mean((mω .- dataset[i]["ωs"]) .^2)
        if mse_ω > 1.0
            sum += 1
            println(i)
        end
    catch e
    end
end


FE = zeros(100)
plot()
for i in 1:n_datasets
    fe = results[i]["fe"] ./ n_samples
    FE += fe
    plot!(fe[3:end], legend=false, linewidth=0.05, color=:black)
end
FE ./= (n_datasets)
plot(FE[3:end], legend=:false, linewidth=3.0, color=:red, xlabel="iteration #", ylabel="Free Energy [nats]")
savefig("figures/FE_analytic.pdf")

using JLD
JLD.save("dump/results_verification_analytic_misture.jld","results",results)
#JLD.save("dump/results_verification_analytic_gates.jld","results",results)

using SparseArrays
resultsJLD = JLD.load("dump/results_verification_analytic_mixture.jld")
results = resultsJLD["results"]
sum_fe = zeros(50)
