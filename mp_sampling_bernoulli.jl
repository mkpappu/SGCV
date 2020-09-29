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

ω1, ω2 = -10.0, -2.0
@RV κ ~ GaussianMeanPrecision(1.0, huge)

@RV s_t ~ Beta(placeholder(:alpha), placeholder(:beta))
@RV b_t ~ Bernoulli(s_t)

f(b) = b*ω1 + (1-b)*ω2
@RV ω ~ Nonlinear{Sampling}(b_t,g=f)
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
m_x = Vector{Float64}(undef, n_samples)
w_x = Vector{Float64}(undef, n_samples)
m_s = Vector{Float64}(undef, n_samples)
a_t = Vector{Float64}(undef, n_samples)
b_t = Vector{Float64}(undef, n_samples)

m_z_t_min, w_z_t_min = 0.0, 10.0
m_x_t_min, w_x_t_min = 0.0, 10.0
a_t_min, b_t_min = 1.0, 1.0

n_its = 10
marginals_mf = Dict()
F_mf = zeros(n_its,n_samples)
@showprogress for t in 1:n_samples
    global m_z_t_min, w_z_t_min, m_x_t_min, w_x_t_min, a_t_min, b_t_min
    # Prepare data and prior statistics
    data = Dict(:y_t      => obs[t],
                :mz_t_min => m_z_t_min,
                :wz_t_min => w_z_t_min,
                :mx_t_min => m_x_t_min,
                :wx_t_min => w_x_t_min,
                :alpha => a_t_min, :beta => b_t_min)

    # Initial recognition distributions
    marginals_mf[:z_t] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
    marginals_mf[:z_t_min] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
    marginals_mf[:b_t] = ProbabilityDistribution(Univariate, Bernoulli, p=0.5)
    marginals_mf[:s_t] = ProbabilityDistribution(Univariate, Beta, a=a_t_min, b=b_t_min)
    marginals_mf[:ω] = vague(SampleList)
    marginals_mf[:κ] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=1.0, w=huge)

    # Execute algorithm
    for i = 1:n_its
        stepX!(data, marginals_mf)
        stepZ!(data, marginals_mf)
        #stepΚ!(data, marginals_mf)
        stepS!(data, marginals_mf)
        stepΩ!(data, marginals_mf)
        F_mf[i,t] = freeEnergy(data, marginals_mf)
    end
    m_z_t_min = ForneyLab.unsafeMean(marginals_mf[:z_t])
    w_z_t_min = ForneyLab.unsafePrecision(marginals_mf[:z_t])
    m_x_t_min = ForneyLab.unsafeMean(marginals_mf[:x_t])
    w_x_t_min = ForneyLab.unsafePrecision(marginals_mf[:x_t])
    a_t_min, b_t_min = marginals_mf[:s_t].params[:a], marginals_mf[:s_t].params[:b]

    m_s_t = ForneyLab.unsafeMean(marginals_mf[:s_t])
    m_y_t = obs[t]

    # Store to buffer
    m_z[t] = m_z_t_min
    w_z[t] = w_z_t_min
    m_x[t] = m_x_t_min
    w_x[t] = w_x_t_min
    m_s[t] = m_s_t
    a_t[t] = a_t_min
    b_t[t] = b_t_min
end


plot(m_z, ribbon=sqrt.(inv.(w_z)))
plot!(z)
plot!(std_x)

m_switches = [x[2] for x in findmax.(m_s)]
scatter(m_switches)
scatter!(switches)

plot(m_x, ribbon=sqrt.(inv.(w_x)))
plot!(x)
scatter!(obs)
