using Random
using Statistics

function generate_swtiching_hgf(n_samples, switches, ωs, variance)
    κs = ones(length(ωs))
    z = Vector{Float64}(undef, n_samples)
    x = Vector{Float64}(undef, n_samples)
    z[1] = 0.0
    x[1] = 0.0
    std_x = []

    stationary = false
    # while !stationary
        for i in 2:n_samples
            # dirty mofo
            if typeof(variance) == Float64
                z[i] = z[i - 1] + sqrt(variance)*randn()
            else
                z[i-1] = variance[i]
            end
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

    # end
    return x, std_x, z

end



function generate_ω(num_ω)
    ωs = [rand(collect(-9:-2))]
    step = rand([2, 3, 4])
    for i in 2:num_ω
        push!(ωs, ωs[end] + step)
    end
    return ωs
end

generate_mnv(dB, ωs) = exp(minimum(ωs))/(10^(dB/10))

function generate_switches(n_switches, n_cats, n_samples, sequences)
    d = Dict(zip(collect(1:n_cats), ones(n_cats)))

    switches = Array{Int64}(undef,n_samples)

    sequence = sequences[rand(collect(1:length(sequences)))]
    switches[1:Int(round(n_samples/n_switches))] .= sequence[1]
    switches[Int(round(n_samples/n_switches))+1:2*Int(round(n_samples/n_switches))] .= sequence[2]
    switches[2*Int(round(n_samples/n_switches))+1:3*Int(round(n_samples/n_switches))] .= sequence[3]
    switches[3*Int(round(n_samples/n_switches))+1:n_samples] .= sequence[4];
    return  switches
end

sequences2 = [[1, 1, 2, 2], [1, 2, 2, 2],  [1, 1, 1, 2],
             [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 1, 2],
             [1, 2, 2, 1], [1, 1, 2, 1], [2, 2, 2, 1]]

sequences1 = [[1, 1, 2, 3], [1, 2, 2, 3], [1, 3, 2, 2], [1, 3, 2, 3],
              [2, 1, 1,  3], [2, 2, 3, 1], [2, 3, 1, 2], [2, 3, 1, 3],
              [3, 2, 2, 1], [3, 3, 1, 2], [3, 1, 2, 3], [3, 2, 3, 1]]

Random.seed!(42)
n_datasets = 10
n_samples = 100
n_cats2 = 2
n_cats1 = 3
dB = 30.0
n_switches = 4
dataset = Dict()
for i in 1:n_datasets
    local switches, reals, omegas, mnv, upper_rw, std_x, obs
    switches2 = generate_switches(n_switches, n_cats2, n_samples, sequences2)
    switches1 = generate_switches(n_switches, n_cats1, n_samples, sequences1)
    omegas2 = generate_ω(n_cats2)
    omegas1 = generate_ω(n_cats1)
    mnv = generate_mnv(dB, omegas1)
    variance = 0.001
    z2, std_z1, upper_rw2 = generate_swtiching_hgf(n_samples, switches2, omegas2, variance)
    reals, std_x, upper_rw1 = generate_swtiching_hgf(n_samples, switches1, omegas1, z2)

    obs = reals .+ sqrt(mnv)*randn(length(reals))

    dataset[i] = Dict("switches2" => switches2,"switches1" => switches1, "ωs1"=> omegas1,"ωs2" => omegas2, "nv" => mnv,
                      "z2" => z2, "std_z1" => std_x, "rw2" => upper_rw2,
                      "reals" => reals, "std_x" => std_x, "rw1" => upper_rw1,
                      "obs" => obs, "var_top" => variance)
end
