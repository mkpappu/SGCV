using ForneyLab
using ProgressMeter
using GCV
using Plots
using SparseArrays
using Random

Random.seed!(42)

n_samples = 100
switches = Array{Int64}(undef,n_samples)
switches[1:Int(round(n_samples/3))] .= 1;
switches[Int(round(n_samples/3))+1:2*Int(round(n_samples/3))] .= 2;
switches[2*Int(round(n_samples/3))+1:n_samples] .= 1;

function generate_swtiching_hgf(n_samples, switches)
    κs = [1.0, 1.0]
    ωs = [-2.0, 2.0]
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

ω1, ω2 = -2.0, 2.0

@RV s_t_min ~ Categorical(placeholder(:m_s_t_min, dims=(2, )))
@RV A ~ Dirichlet(placeholder(:A_t_min, dims=(2, 2)))
@RV s_t ~ Transition(s_t_min, A)

f(s) = s[1]*ω1 + s[2]*ω2
@RV ω ~ Nonlinear{Sampling}(s_t,g=f, n_samples=1000)
@RV z_t_min ~ GaussianMeanPrecision(placeholder(:mz_t_min),placeholder(:wz_t_min))
@RV z_t ~ GaussianMeanPrecision(z_t_min, 100.0)
@RV x_t_min ~ GaussianMeanPrecision(placeholder(:mx_t_min),placeholder(:wx_t_min))
@RV x_t ~ GaussianControlledVariance(x_t_min, z_t, 1.0, ω)
@RV y_t ~ GaussianMeanPrecision(x_t, 10.0)
# Data placeholder
placeholder(y_t, :y_t)

q = PosteriorFactorization([z_t_min;z_t],[x_t_min;x_t], A, s_t, s_t_min; ids=[:Z :X :A :S :S0])
algo = messagePassingAlgorithm(free_energy=true)
src_code = algorithmSourceCode(algo, free_energy=true)

eval(Meta.parse(src_code))

# Define values for prior statistics

m_z = Vector{Float64}(undef, n_samples)
w_z = Vector{Float64}(undef, n_samples)
m_x = Vector{Float64}(undef, n_samples)
w_x = Vector{Float64}(undef, n_samples)
m_s = Vector{Array{Float64}}(undef, n_samples)
m_A = Vector{Matrix}(undef, n_samples)
s_ω = Vector{Array{Float64}}(undef, n_samples)
w_ω = Vector{Array{Float64}}(undef, n_samples)

m_z_t_min, w_z_t_min = 0.0, 10.0
s_ω_min, w_ω_min = vague(SampleList).params[:s], vague(SampleList).params[:w]
m_x_t_min, w_x_t_min = 0.0, 1.0
m_s_min = Array([0.5, 0.5])
A_min = [100 1; 1 100]

n_its = 50
marginals_mf = Dict()
F_mf = zeros(n_its,n_samples)
@showprogress for t in 1:n_samples
    global m_z_t_min, w_z_t_min, m_x_t_min, w_x_t_min, m_s_min, A_min, s_ω_min, w_ω_min
    # Prepare data and prior statistics
    data = Dict(:y_t       => obs[t],
                :mz_t_min => m_z_t_min,
                :wz_t_min => w_z_t_min,
                :mx_t_min => m_x_t_min,
                :wx_t_min => w_x_t_min,
                :A_t_min => A_min,
                :m_s_t_min => m_s_min)

    # Initial recognition distributions
    marginals_mf[:z_t] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=m_z_t_min, w=w_z_t_min)
    marginals_mf[:z_t_min] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=m_z_t_min, w=w_z_t_min)
    marginals_mf[:s_t_min] = ProbabilityDistribution(Categorical, p=Array(m_s_min))
    marginals_mf[:s_t] = ProbabilityDistribution(Categorical, p=Array(m_s_min))
    marginals_mf[:A] = ProbabilityDistribution(MatrixVariate, Dirichlet, a=A_min)
    marginals_mf[:ω] = ProbabilityDistribution(Univariate, SampleList, s=s_ω_min, w=w_ω_min)
    marginals_mf[:κ] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=1.0, w=huge)

    # Execute algorithm
    for i = 1:n_its
        stepX!(data, marginals_mf)
        #stepΚ!(data, marginals_mf)
        stepA!(data, marginals_mf)
        stepS0!(data, marginals_mf)
        stepS!(data, marginals_mf)
        stepZ!(data, marginals_mf)
        #stepΩ!(data, marginals_mf)
        F_mf[i,t] = freeEnergy(data, marginals_mf)
    end
    m_z_t_min = ForneyLab.unsafeMean(marginals_mf[:z_t])
    w_z_t_min = ForneyLab.unsafePrecision(marginals_mf[:z_t])
    m_x_t_min = ForneyLab.unsafeMean(marginals_mf[:x_t])
    w_x_t_min = ForneyLab.unsafePrecision(marginals_mf[:x_t])
    m_s_t = ForneyLab.unsafeMean(marginals_mf[:s_t])
    A_min = ForneyLab.unsafeMean(marginals_mf[:A])
    s_ω_min = marginals_mf[:ω].params[:s]
    w_ω_min = marginals_mf[:ω].params[:w]

    m_y_t = obs[t]

    # Store to buffer
    m_x[t] = m_x_t_min
    w_x[t] = w_x_t_min
    m_z[t] = m_z_t_min
    w_z[t] = w_z_t_min
    m_s[t] = m_s_t
    m_A[t] = A_min
    s_ω[t] = s_ω_min
    w_ω[t] = w_ω_min

end

plot(m_z, ribbon=sqrt.(inv.(w_z)))
plot!(z)
#plot!(std_x)

m_switches = [x[2] for x in findmax.(m_s)]
scatter(m_switches)
scatter!(switches)

plot(m_x, ribbon=sqrt.(inv.(w_x)))
plot!(x)
scatter!(obs)
