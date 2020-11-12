function generate_swtiching_hgf(n_samples, switches, ωs)
    κs = ones(length(omegas))
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
        elseif switches[i] == 3
            push!(std_x, sqrt(exp(κs[3]*z[i] + ωs[3])))
            x[i] = x[i - 1] + std_x[end]*randn()
        end
    end
    return x, std_x, z
end

# NOTE: Ask Ismail

Random.seed!(100)

mnv = 0.01
omegas = [-2.0, 2.0, 5.0]
dims = length(omegas)

n_samples = 100
switches = Array{Int64}(undef,n_samples)
switches[1:Int(round(n_samples/3))] .= 1;
switches[Int(round(n_samples/3))+1:2*Int(round(n_samples/3))] .= 2;
switches[2*Int(round(n_samples/3))+1:n_samples] .= 3;


reals, std_x, upper_rw = generate_swtiching_hgf(n_samples, switches, omegas)
obs = reals .+ sqrt(mnv)*randn(length(reals))
