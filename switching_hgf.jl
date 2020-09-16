using Turing
using Plots

function generate_swtiching_hgf(n_samples)
    κ = [0.1, 2.0]
    ω = [1.0, -2.0]
    z = Vector{Float64}(undef, n_samples)
    x = Vector{Float64}(undef, n_samples)
    s = Vector{Int64}(undef, n_samples)
    A = [0.95 0.05; 0.05 0.95]

    z[1] = 0.0
    x[1] = 0.0
    s[1] = rand(Categorical([0.5, 0.5]))
    rw = 0.01
    
    std_x = []

    for i in 2:n_samples
        z[i] = z[i - 1] + sqrt(rw)*randn()
        s[i] = rand(Categorical((A[:, s[i-1]])))
        push!(std_x, sqrt(exp(κ[s[i]]*z[i] + ω[s[i]])))
        x[i] = x[i - 1] + std_x[end]*randn()
    end
    return x, std_x, s
end


@model switching_gcv(y) = begin
    K = 2
    n_samples = length(y)
    z = Vector(undef, n_samples+1)
    x = Vector(undef, n_samples+1)
    s = Vector(undef, n_samples+1)
    A = Vector{Vector}(undef, K)
    κ ~ MvNormal([0.1, 2.0], 1.0)
    ω ~ MvNormal([1.0, -2.0], 1.0)
    s[1] ~ Categorical([0.5, 0.5])
    z[1] ~ Normal(0.0, 1.0)
    x[1] ~ Normal(0.0, 1.0)

    for i in 1:K
        A[i] ~ Dirichlet(ones(K)/K)
    end

    for i in 2:n_samples+1
        s[i] ~ Categorical(vec(A[s[i-1]]))
        z[i] ~ Normal(z[i-1], 0.1)
        x[i] ~ Normal(x[i-1], sqrt(exp(κ[s[i]]*z[i] + ω[s[i]])))
        y[i-1] ~ Normal(x[i], 1.0)

    end
end

using JLD
n_samples = 99

state, variance, category = generate_swtiching_hgf(100)
observations = state .+ randn(100)

save("/Users/albertpod/Desktop/dump.jld", "state", state, "variance", variance, "category", category, "observations", observations)

smodel = switching_gcv(observations)

chain = sample(smodel, SMC(), 100)

samples = get(chain,:x)
mx = [mean(samples.x[i].data) for i in 1:n_samples]
vx = [std(samples.x[i].data) for i in 1:n_samples]

samples = get(chain,:z)
mz = [mean(samples.z[i].data) for i in 1:n_samples]
vz = [std(samples.z[i].data) for i in 1:n_samples]

samples = get(chain,:s)
ms = [mean(samples.s[i].data) for i in 1:n_samples]
