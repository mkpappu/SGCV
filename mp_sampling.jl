using ForneyLab
using GCV
using ProgressMeter
using Plots

n_samples = 100
switches = Array{Int64}(undef,n_samples)
switches[1:25] .= 1;
switches[26:60] .= 2;
switches[61:n_samples] .= 1;

function generate_swtiching_hgf(n_samples, switches)
    κs = [1.0, 1.0]
    ωs = [-10.0, -2.0]
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
obs = x .+ sqrt(0.1)*randn(length(x))

fg = FactorGraph()

κ1, κ2 = 2.0, 0.5
@RV ω ~ GaussianMeanPrecision(1.0, huge)

@RV s_t ~ Beta(placeholder(:alpha), placeholder(:beta))
@RV b_t ~ Bernoulli(s_t)

f(b) = b*κ1 + (1-b)*κ2
@RV κ ~ Nonlinear{Sampling}(b_t,g=f)
@RV z_t_min ~ GaussianMeanPrecision(placeholder(:mz_t_min),placeholder(:wz_t_min))
@RV z_t ~ GaussianMeanPrecision(z_t_min, 10.0)
@RV x_t_min ~ GaussianMeanPrecision(placeholder(:mx_t_min),placeholder(:wx_t_min))
@RV x_t ~ GaussianControlledVariance(x_t_min, z_t, κ, ω)
@RV y_t ~ GaussianMeanPrecision(x_t, 1.0)
# Data placeholder
placeholder(y_t, :y_t)

q = PosteriorFactorization([z_t_min;z_t],[x_t_min;x_t], κ, ω, s_t; ids=[:Z :X :Κ :Ω :S])
algo = messagePassingAlgorithm(free_energy=true)
src_code = algorithmSourceCode(algo, free_energy=true)

eval(Meta.parse(src_code))

# Define values for prior statistics

m_z = Vector{Float64}(undef, n_samples)
w_z = Vector{Float64}(undef, n_samples)
m_s = Vector{Float64}(undef, n_samples)

m_z_t_min, w_z_t_min = 0.0, 10.0
m_x_t_min, w_x_t_min = 0.0, 10.0

n_its = 10
marginals_mf = Dict()
F_mf = zeros(n_its,n_samples)
@showprogress for t in 1:n_samples
    global m_z_t_min, w_z_t_min, m_x_t_min, w_x_t_min
    # Prepare data and prior statistics
    data = Dict(:y_t       => obs[t],
                :mz_t_min => m_z_t_min,
                :wz_t_min => w_z_t_min,
                :mx_t_min => m_x_t_min,
                :wx_t_min => w_x_t_min,
                :alpha => 3.0, :beta => 1.0)

    # Initial recognition distributions
    marginals_mf[:z_t] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
    marginals_mf[:z_t_min] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
    marginals_mf[:b_t] = ProbabilityDistribution(Univariate, Bernoulli, p=0.5)
    marginals_mf[:s_t] = vague(Beta)
    marginals_mf[:κ] = vague(SampleList)
    marginals_mf[:ω] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=1.0, w=huge)

    # Execute algorithm
    for i = 1:n_its
        stepX!(data, marginals_mf)
        stepZ!(data, marginals_mf)
        stepΚ!(data, marginals_mf)
        stepS!(data, marginals_mf)
        #stepΩ!(data, marginals_mf)
        F_mf[i,t] = freeEnergy(data, marginals_mf)
    end
    m_z_t_min = ForneyLab.unsafeMean(marginals_mf[:z_t])
    w_z_t_min = ForneyLab.unsafePrecision(marginals_mf[:z_t])
    m_s_t = ForneyLab.unsafeMean(marginals_mf[:s_t])
    m_y_t = y_data[t]

    # Store to buffer
    m_z[t] = m_z_t_min
    w_z[t] = w_z_t_min
    m_s[t] = m_s_t
end

function stepX1!(data::Dict, marginals::Dict=Dict(), messages::Vector{Message}=Array{Message}(undef, 4))

    messages[1] = ruleVBGaussianMeanPrecisionOut(nothing, ProbabilityDistribution(Univariate, PointMass, m=data[:mx_t_min]), ProbabilityDistribution(Univariate, PointMass, m=data[:wx_t_min]))
    messages[2] = ruleVBGaussianMeanPrecisionM(ProbabilityDistribution(Univariate, PointMass, m=data[:y_t]), nothing, ProbabilityDistribution(Univariate, PointMass, m=1.0))
    messages[3] = ruleSVBGaussianControlledVarianceXGNDDD(messages[2], nothing, marginals[:z_t], marginals[:κ], marginals[:ω])
    messages[4] = ruleSVBGaussianControlledVarianceOutNGDDD(nothing, messages[1], marginals[:z_t], marginals[:κ], marginals[:ω])

    marginals[:x_t] = messages[4].dist * messages[2].dist
    marginals[:x_t_min] = messages[1].dist * messages[3].dist
    marginals[:x_t_x_t_min] = ruleMGaussianControlledVarianceGGDDD(messages[2], messages[1], marginals[:z_t], marginals[:κ], marginals[:ω])

    return marginals

end
