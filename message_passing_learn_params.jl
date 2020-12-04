using ForneyLab, GCV, ProgressMeter, DelimitedFiles

pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,3,'0')) # Left-pads a number with zeros, converts it to symbol and appends to sym

n_samples = 4
n_layers = 2
n_its = 5
n_dataset = 1
omega  = -3.0
kappa = 1.0
var_top = 0.01
var_obs = 0.1
obs = readdlm("/Users/isenoz/Documents/Phd/ThirdYear/sampling_scripts_simulations/data/simple_2_layer/observations.txt")
x_golden = readdlm("/Users/isenoz/Documents/Phd/ThirdYear/sampling_scripts_simulations/data/simple_2_layer/x_golden.txt")
z_golden = readdlm("/Users/isenoz/Documents/Phd/ThirdYear/sampling_scripts_simulations/data/simple_2_layer/z_golden.txt")

# 1 layer
function generate_mp()
    fg = FactorGraph()

    @RV [id=:kappa] kappa ~ GaussianMeanVariance(placeholder(:mk),placeholder(:vk))
    @RV [id=:omega] omega ~ GaussianMeanVariance(placeholder(:mo),placeholder(:vo))
    @RV [id=pad(:x,0)] x_0 ~ GaussianMeanPrecision(placeholder(:mx_0),placeholder(:wx_0)) # Prior
    @RV [id=pad(:z,0)] z_0 ~ GaussianMeanPrecision(placeholder(:mz_0),placeholder(:wz_0))
    z = Vector{Variable}(undef,n_samples)
    x = Vector{Variable}(undef,n_samples) # Pre-define vectors for storing latent and observed variables
    y = Vector{Variable}(undef,n_samples)
    z_t_prev = z_0
    x_t_prev = x_0
    for t = 1:n_samples
        @RV [id=pad(:z,t)] z[t] ~ GaussianMeanPrecision(z_t_prev, placeholder(pad(:prc_trns,t)))
        @RV [id=pad(:x,t)] x[t] ~ GaussianControlledVariance(x_t_prev,z[t],kappa,omega) # Process model
        @RV [id=pad(:y,t)] y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:prc_llh,t))) # Observation model

        placeholder(y[t], :y, index=t) # Indicate observed variable

        x_t_prev = x[t] # Prepare state for next section
        z_t_prev = z[t]
    end
    ;

    algo = variationalAlgorithm([z_0;z],[x_0;x],kappa,omega,ids=[:Z,:X, :K, :Ω],free_energy=true)
    src_code = algorithmSourceCode(algo, free_energy=true);

    return src_code
end

function mp(obs;var_top,var_obs,mk,vk,mo,vo)
    mx = Array{Float64}(undef,n_samples+1);vx = Array{Float64}(undef,n_samples+1)
    mz = Array{Float64}(undef,n_samples+1);vz = Array{Float64}(undef,n_samples+1)

    fe = Array{Float64}(undef,n_its)
    fe_nuts = Array{Float64}(undef,n_its)


    global marginals = Dict()
    global data_dict = Dict()
    data_dict[:y] =  obs
    marginals[pad(:z,n_samples)] = vague(GaussianMeanPrecision)
    marginals[pad(:x,n_samples)] = vague(GaussianMeanPrecision)
    marginals[:omega] = vague(GaussianMeanVariance)
    marginals[:kappa] = vague(GaussianMeanVariance)
    for t = 1:n_samples
        global data_dict[pad(:prc_trns,t)]= 1/(var_top)
                data_dict[:mk] = mk
                data_dict[:mo] = mo
                data_dict[:vk] = vk
                data_dict[:vo] = vo
                data_dict[pad(:omega,t)] = omega
                data_dict[pad(:prc_llh,t)] = 1/(var_obs)
        marginals[pad(:z,t)] = ProbabilityDistribution(GaussianMeanPrecision,m=0.0,w=0.1)
        marginals[pad(:x,t)] = ProbabilityDistribution(GaussianMeanPrecision,m=0.0,w=0.1)
        marginals[pad(:z,t)*:_*pad(:z,t-1)] = ProbabilityDistribution(Multivariate,GaussianMeanPrecision,m=zeros(2),w=0.01*diageye(2))
        marginals[pad(:x,t)*:_*pad(:x,t-1)] = ProbabilityDistribution(Multivariate,GaussianMeanPrecision,m=zeros(2),w=0.01*diageye(2))
    end


    for i=1:n_its
        stepX!(data_dict,marginals)
        stepZ!(data_dict,marginals)
        stepΩ!(data_dict, marginals)
        stepK!(data_dict, marginals)

        fe[n,i] = freeEnergy(data_dict,marginals)        # fe_nuts[n,i] = freeEnergy(data_dict,marginals_nuts)

    end

        mz[n,:] = [mean(marginals[pad(:z,t)]) for t=0:n_samples]
        vz[n,:] = [var(marginals[pad(:z,t)]) for t=0:n_samples]
        mx[n,:] = [mean(marginals[pad(:x,t)]) for t=0:n_samples]
        vx[n,:] = [var(marginals[pad(:x,t)]) for t=0:n_samples];
        mω = ForneyLab.unsafeMean(marginals[:omega])
        mκ= ForneyLab.unsafeMean(marginals[:kappa])
    return mx,vx,mz,vz,mκ,mω,fe
end

src_code = generate_mp()
eval(Meta.parse(src_code));

mx,vx,mz,vz,mκ,mω,fe = mp(obs=, 0.01,0.1,1.0,0.001,0.0,5.0);
