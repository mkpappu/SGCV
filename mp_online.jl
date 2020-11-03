include("SGCV.jl")
using Revise
using Main.SGCV
using ForneyLab
using LinearAlgebra
using Plots
using ProgressMeter
using LinearAlgebra

n_samples = 200

switches = Array{Int64}(undef,n_samples)
switches[1:20] .= 2;
switches[21:50] .= 1;
switches[51:150] .=2
switches[151:170] .= 1;
switches[171:end] .= 2;
# switches[1:end] .= 1;

function generate_swtiching_hgf(n_samples, switches)
    κs = [1.0, 1.0]
    ωs = [5.0, -2.0]
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

x, std_x, z = generate_swtiching_hgf(n_samples, switches)
obs = x .+ sqrt(0.01)*randn(length(x))

function generate_algorithm(ndim, n_samples)
    fg = FactorGraph()
    @RV A ~ Dirichlet(placeholder(:A_prior,dims=(ndim,ndim)))
    @RV z_min ~ GaussianMeanPrecision(placeholder(:mz_prior), placeholder(:wz_prior))
    @RV x_min ~ GaussianMeanPrecision(placeholder(:mx_prior), placeholder(:wx_prior))
    @RV s_min ~ ForneyLab.Categorical(placeholder(:s_prior,dims=(ndim,)))


    @RV s ~ Transition(s_min, A)
    @RV z ~ GaussianMeanPrecision(z_min, placeholder(:wz_transition))
    @RV x ~ SwitchingGaussianControlledVariance(x_min, z, placeholder(:k,dims=(ndim,)),placeholder(:w,dims=(ndim,)),s)
    @RV y ~ GaussianMeanPrecision(x, placeholder(:wy_transition))
    placeholder(y, :y )

    q = PosteriorFactorization([x;x_min], [z;z_min] ,[s;s_min], A, ids=[:X :Z :S :A])
    algo = messagePassingAlgorithm(free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);
    return src_code
end

code = generate_algorithm(2, n_samples)
eval(Meta.parse(code))

function infer(obs;
    ndims = 2,
    κ_m_prior = [1.0, 1.0],
    ω_m_prior = [5.0, -2.0],
    κ_w_prior =  huge .* diageye(ndims),
    ω_w_prior = huge .* diageye(ndims),
    z_m_prior = 0.0,
    z_w_prior = 100.0,
    x_m_prior = 0.0,
    x_w_prior = 0.1,
    x_x_m_prior = zeros(ndims),
    x_x_w_prior = 0.1*diageye(ndims),
    z_z_m_prior = zeros(ndims),
    z_z_w_prior = 100.0*diageye(ndims),
    z_w_transition_prior = 100.0,
    y_w_transition_prior =  1/0.01,
    A_prior = ones(ndims,ndims),
    s_prior = ones(ndims) ./ ndims
)

    marginals = Dict()
    marginals[:A] = ProbabilityDistribution(ForneyLab.MatrixVariate, Dirichlet, a=ones(2, 2))#vague(Dirichlet, (2, 2))
    marginals[:z] = ProbabilityDistribution(ForneyLab.Univariate, GaussianMeanPrecision, m = z_m_prior, w = z_w_prior)
    marginals[:s] = ProbabilityDistribution(Categorical, p = [0.5, 0.5])
    marginals[:s_s_min] = ProbabilityDistribution(Contingency,p=[0.5 0.5; 0.5 0.5])
    marginals[:z_z_min] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = z_z_m_prior, w = z_z_w_prior)
    marginals[:x_x_min] = ProbabilityDistribution(ForneyLab.Multivariate,GaussianMeanPrecision, m = x_x_m_prior, w = x_x_w_prior)

    data = Dict()

    n_its = 10
    fe = Matrix{Float64}(undef, n_samples, n_its)
    mz = Vector{Float64}(undef, n_samples)
    vz = Vector{Float64}(undef, n_samples)
    mx = Vector{Float64}(undef, n_samples)
    vx = Vector{Float64}(undef, n_samples)
    ms = Vector{Any}(undef, n_samples)
    mA = Vector{Any}(undef, n_samples)
    data[:k] = κ_m_prior
    data[:w] = ω_m_prior
    data[:wz_transition] = z_w_transition_prior
    data[:wy_transition] = y_w_transition_prior
    ##
    @showprogress "Time" for t = 1:n_samples
        data[:y] = obs[t]
        data[:mz_prior] = z_m_prior
        data[:wz_prior] = z_w_prior
        data[:mx_prior] = x_m_prior
        data[:wx_prior] = x_w_prior
        data[:A_prior] = A_prior
        data[:s_prior] = s_prior


        for i = 1:n_its
            stepX!(data, marginals)
            stepA!(data, marginals)
            stepS!(data, marginals)
            stepZ!(data, marginals)

            fe[t,i] = freeEnergy(data, marginals)
        end

        mz[t],vz[t] = ForneyLab.unsafeMeanCov(marginals[:z])
        mx[t],vx[t] = ForneyLab.unsafeMeanCov(marginals[:x])
        ms[t] = ForneyLab.unsafeMean(marginals[:s])
        mA[t] = ForneyLab.unsafeMean(marginals[:A])

        (z_m_prior,z_w_prior) = (mz[t],inv(vz[t]))
        (x_m_prior,x_w_prior) = (mx[t],inv(vx[t]))
        s_prior = ms[t]
        A_prior = mA[t]

    end

    return mz,vz,mx,vx,ms,mA,fe
end

mz,vz,mx,vx,ms,mA,fe = infer(obs)

plot(mx[1:300], ribbon=sqrt.(vx[1:300]))
plot!(x[1:300])
scatter!(obs[1:300])

p3 = plot(mz, ribbon=sqrt.(vz))
p3 = plot!(z)

estimate = zeros(20,n_samples)
for t=1:n_samples
    for i=1:10
        estimate[i,t] = ms[t][1]
        estimate[i+10,t] = ms[t][2]
    end
end
l = @layout [a ; b]
p1 = scatter(switches)
p2 = plot(Gray.(estimate), link=:x, xlabel="Time")
plot(p1,p2,layout = l)

plot(sum(fe,dims=1)')
plot(sum(fe,dims=2))
