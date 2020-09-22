using Turing
using Plots

n_samples = 100
switches = Array{Int64}(undef,n_samples)
switches[1:25] .= 1;
switches[26:60] .= 2;
switches[61:n_samples] .= 1;
# scatter(switches)

function generate_swtiching_hgf(n_samples, switches, κs, ωs)
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

κs = [1.0, 1.0]
ωs = [-10.0, -2.0]
x, std_x, z = generate_swtiching_hgf(n_samples, switches, κs, ωs)
obs = x .+ 0.1 .* randn(length(x))

@model SHGF(y, κs, ωs) = begin
    n_samples = length(y)
    dims = length(κs)
    # Transition matrix.
    A = Vector{Vector}(undef, dims)
    z = tzeros(Real, n_samples + 1)
    s = tzeros(Int, n_samples + 1)
    x = tzeros(Real, n_samples + 1)
    z[1] ~ Normal(0.0, 1.0)
    x[1] ~ Normal(0.0, 1.0)
    s[1] ~ Categorical(dims)

    for i in 1:dims
        A[i] ~ Dirichlet(ones(dims)/dims)
    end

    for t in 2:n_samples + 1
        s[t] ~ Categorical(vec(A[s[t-1]]))
        z[t] ~ Normal(z[t-1], 0.1)
        x[t] ~ Normal(x[t-1], exp(κs[s[t]]*z[t] + ωs[s[t]]))
        y[t-1] ~ Normal(x[t], 0.1)
    end
end


g = Gibbs(HMC(0.001, 7, :x, :z, :A), PG(20, :s))
nuts = NUTS(0.69)
hmc = HMC(0.001, 5)
pf = SMC(1000)

chn = sample(SHGF(obs, κs, ωs), PG(10), 1000)

samples = get(chn, :x)
mx = [mean(samples.x[i].data) for i in 1:n_samples]
vx = [std(samples.x[i].data) for i in 1:n_samples]

samples = get(chn, :z)
mz = [mean(samples.z[i].data) for i in 1:n_samples]
vz = [std(samples.z[i].data) for i in 1:n_samples]

samples = get(chn, :s)
ms = [mean(samples.s[i].data) for i in 1:n_samples]
ms = Int64.(round.(ms))

plot(x)
plot!(mx, ribbon=vx)
scatter!(obs)

scatter(switches)
scatter!(ms)

samples = get(chn, :A)
A_est = [mean(samples.A[1]) mean(samples.A[2]); mean(samples.A[3]) mean(samples.A[4])]