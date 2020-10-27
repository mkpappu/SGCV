import ForneyLab: collectStructuredVariationalNodeInbounds, ultimatePartner, current_inference_algorithm,
                  posteriorFactor,localEdgeToRegion, ultimatePartner, assembleClamp!,posteriorFactor,isClamped,
                  @symmetrical

export ruleSVBSwitchingGaussianControlledVarianceIn1MPPPP,
       ruleSVBSwitchingGaussianControlledVarianceMIn2PPPP,
       ruleSVBSwitchingGaussianControlledVariancePIn3PPP,
       ruleSVBSwitchingGaussianControlledVariancePPIn4PP,
       ruleSVBSwitchingGaussianControlledVariancePPPIn5P,
       ruleSVBSwitchingGaussianControlledVariancePPPPIn6,
       ruleMSwitchingGaussianControlledVariance,
       prod!,
       ruleSVBGaussianMeanPrecisionMFND,
       ruleMGaussianMeanPrecisionFGD




include("approximations/ghcubature.jl")
include("approximations/srcubature.jl")

category(p) = findmax(p)[2]

function ϕ(z, κ, ω, s)
    ms = unsafeMean(s)
    mω, Vω = unsafeMeanCov(ω)
    mz, vz = unsafeMeanCov(z)
    mκ, Vκ = unsafeMeanCov(κ)
    select = category(ms)
    exp(-mκ[select]*mz - mω[select] + 0.5((mκ[select])^2*vz + mz^2*Vκ[select,select] + Vκ[select,select]*vz + Vω[select]))
end

function ψ(yx)
    m, V = unsafeMeanCov(yx)
    (m[1] - m[2])*(m[1] - m[2]) + V[1] + V[4] - V[3] - V[2]
end

function ruleSVBSwitchingGaussianControlledVarianceIn1MPPPP(msg_in1::Nothing, msg_in2::Message{F},
            dist_in3::ProbabilityDistribution, dist_in4::ProbabilityDistribution,
            dist_in5::ProbabilityDistribution, dist_in6::ProbabilityDistribution) where F<:Gaussian

    Message(Univariate, GaussianMeanVariance, m=unsafeMean(msg_in2.dist), v=unsafeCov(msg_in2.dist) + 1/ϕ(dist_in3, dist_in4, dist_in5, dist_in6))

end

ruleSVBSwitchingGaussianControlledVarianceMIn2PPPP(msg_in1::Message{F}, msg_in2::Nothing,
            dist_in3::ProbabilityDistribution, dist_in4::ProbabilityDistribution,
            dist_in5::ProbabilityDistribution, dist_in6::ProbabilityDistribution) where F<:Gaussian = ruleSVBSwitchingGaussianControlledVarianceIn1MPPPP(msg_in2, msg_in1, dist_in3, dist_in4, dist_in5, dist_in6)


function ruleSVBSwitchingGaussianControlledVariancePIn3PPP(dist_in1_in2::ProbabilityDistribution,
            msg_in3::Nothing, dist_in4::ProbabilityDistribution,
            dist_in5::ProbabilityDistribution, dist_in6::ProbabilityDistribution)

    mκ, Vκ = unsafeMeanCov(dist_in4)
    mω, Vω = unsafeMeanCov(dist_in5)
    ms = unsafeMean(dist_in6)
    select = category(ms)
    l_pdf(z) = begin
        -0.5*(mκ[select]*z + ψ(dist_in1_in2)*exp(-mκ[select]*z -mω[select] + 0.5*Vω[select,select]))
    end
    Message(Univariate, Function, log_pdf = l_pdf, cubature = ghcubature(1, 20))

end

function ruleSVBSwitchingGaussianControlledVariancePPIn4PP(dist_in1_in2::ProbabilityDistribution,
            dist_in3::ProbabilityDistribution, msg_in4::Nothing,
            dist_in5::ProbabilityDistribution, dist_in6::ProbabilityDistribution)

    mz, vz = unsafeMeanCov(dist_in3)
    mω, Vω = unsafeMeanCov(dist_in5)
    ms = unsafeMean(dist_in6)

    l_pdf(κ) = begin
        -0.5*(ms'*κ*mz + ψ(dist_in1_in2)*exp(-ms'*κ*mz - ms'*mω + 0.5*ms'*Vω*ms))
    end
    Message(Multivariate, Function, log_pdf = l_pdf, cubature = ghcubature(dims(dist_in5), 20))

end


function ruleSVBSwitchingGaussianControlledVariancePPPIn5P(dist_in1_in2::ProbabilityDistribution,
            dist_in3::ProbabilityDistribution, dist_in4::ProbabilityDistribution,
            msg_in5::Nothing, dist_in6::ProbabilityDistribution)

    mz, vz = unsafeMeanCov(dist_in3)
    mκ, Vκ = unsafeMeanCov(dist_in4)
    ms = unsafeMean(dist_in6)

    l_pdf(ω) = begin
        -0.5*(ms'*ω + ψ(dist_in1_in2)*exp(-ms'*ω))
    end
    Message(Multivariate, Function, log_pdf = l_pdf, cubature = ghcubature(dims(dist_in4), 20))

end


function ruleSVBSwitchingGaussianControlledVariancePPPPIn6(dist_in1_in2::ProbabilityDistribution,
            dist_in3::ProbabilityDistribution, dist_in4::ProbabilityDistribution,
            dist_in5::ProbabilityDistribution, msg_in6::Nothing)

    mz, vz = unsafeMeanCov(dist_in3)
    mκ, Vκ = unsafeMeanCov(dist_in4)
    mω, Vω = unsafeMeanCov(dist_in5)
    A = exp.(-mω+0.5diag(Vω))
    B = exp.(-mκ .* mz .+ 0.5(mκ.^2*vz .+ mz^2 .* diag(Vκ) + diag(Vκ) .* vz))
    r = exp.(-0.5.*(mκ .* mz .+ mω + ψ(dist_in1_in2) .* A .* B))

    Message(Univariate, Categorical, p = r ./ sum(r))
end

function ruleMSwitchingGaussianControlledVariance(msg_in1::Message{F1}, msg_in2::Message{F2},
            dist_in3::ProbabilityDistribution, dist_in4::ProbabilityDistribution,
            dist_in5::ProbabilityDistribution, dist_in6::ProbabilityDistribution) where {F1<:Gaussian, F2<:Gaussian}

    xi_y, w_y = unsafeWeightedMeanPrecision(msg_in1.dist)
    xi_x, w_x = unsafeWeightedMeanPrecision(msg_in2.dist)
    ϕ̂ = ϕ(dist_in3, dist_in4, dist_in5, dist_in6)

    λ = [ϕ̂+w_y -ϕ̂; -ϕ̂ ϕ̂+w_x]
    ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision, xi=[xi_y; xi_x], w=λ)
end

@symmetrical function prod!(x::ProbabilityDistribution{Multivariate, Function},
                            y::ProbabilityDistribution{Multivariate, F},
                            z::ProbabilityDistribution{Multivariate, GaussianMeanVariance}=ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(3), v=diageye(3))) where {F<:Gaussian}
    y = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},y)
    ndims = dims(y)
    g(s) = exp(x.params[:log_pdf](s))
    cubature = x.params[:cubature]
    m, V = approximate_meancov(cubature, g, y)
    z.params[:m] = m
    z.params[:v] = V
    return z
end


@symmetrical function prod!(x::ProbabilityDistribution{Univariate, Function},
                            y::ProbabilityDistribution{Univariate, F},
                            z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=1.0)) where {F<:Gaussian}
    y = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},y)
    ndims = dims(y)
    g(s) = exp(x.params[:log_pdf](s))
    cubature = x.params[:cubature]
    m, V = approximate_meancov(cubature, g, y)
    z.params[:m] = m
    z.params[:v] = V
    return z
end


# extend GaussianMeanPrecision rules

function ruleSVBGaussianMeanPrecisionMFND(msg_out::Message{Function,Multivariate},
                                          msg_mean::Message{F,Multivariate},
                                          dist_prec::ProbabilityDistribution) where F<:Gaussian
    msg_fwd = ruleSVBGaussianMeanPrecisionOutVGD(nothing,msg_mean,dist_prec)
    m_mean,v_mean = unsafeMeanCov(msg_fwd.dist*msg_out.dist)
    return Message(Multivariate,GaussianMeanVariance,m=m_mean,v=v_mean+inv(unsafeMean(dist_prec)))
end

function ruleMGaussianMeanPrecisionFGD(msg_out::Message{Function,Multivariate},
                                       msg_mean::Message{F,Multivariate},
                                       dist_prec::ProbabilityDistribution) where F<:Gaussian
    d = dims(msg_mean.dist)
    m_mean,v_mean = unsafeMeanCov(msg_mean.dist)
    Wbar = unsafeMean(dist_prec)
    jitter = Diagonal(1e-13*(rand(size(Wbar,1))) .* Diagonal(Wbar))
    W = [(Wbar + jitter) -Wbar; -Wbar (Wbar + jitter)]
    # W = [Wbar -Wbar; -Wbar Wbar]
    logpdf = msg_out.dist.params[:log_pdf]
    msg_fwd = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_mean, dist_prec)
    v_mean_inv = cholinv(v_mean)
    l(z) = @views -0.5 * z'*W*z - 0.5 * (z[d+1:end] - m_mean)' * v_mean_inv * (z[d+1:end] - m_mean) + logpdf(z[1:d])
    # Expansion point
    point1 = unsafeMean(msg_fwd.dist * msg_out.dist)
    try
        m_joint, v_joint = NewtonMethod(l, [ point1; m_mean ])
        return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=m_joint,v=v_joint)
    catch e
        # Use cubature as a fallback
        cubature  = msg_out.dist.params[:cubature]
        mean, cov = approximate_meancov(cubature, (x) -> exp(logpdf(x)), msg_fwd.dist)
        approx_message = Message(Multivariate, GaussianMeanVariance, m = mean, v = cov)
        println(approx_message)
        println(msg_mean)
        return ruleMGaussianMeanPrecisionGGD(approx_message, msg_mean, dist_prec)
    end
end


function ruleSVBGaussianMeanPrecisionMFND(msg_out::Message{Function,Univariate},
                                          msg_mean::Message{F,Univariate},
                                          dist_prec::ProbabilityDistribution) where F<:Gaussian
    msg_fwd = ruleSVBGaussianMeanPrecisionOutVGD(nothing,msg_mean,dist_prec)
    m_mean,v_mean = unsafeMeanCov(msg_fwd.dist*msg_out.dist)
    return Message(Univariate,GaussianMeanVariance,m=m_mean,v=v_mean+inv(unsafeMean(dist_prec)))
end

function ruleMGaussianMeanPrecisionFGD(msg_out::Message{Function,Univariate},
                                       msg_mean::Message{F,Univariate},
                                       dist_prec::ProbabilityDistribution) where F<:Gaussian
    d = dims(msg_mean.dist)
    m_mean,v_mean = unsafeMeanCov(msg_mean.dist)
    Wbar = unsafeMean(dist_prec)
    # jitter = Diagonal(1e-13*(rand(size(Wbar,1))) .* Diagonal(Wbar))
    # W = [(Wbar + jitter) -Wbar; -Wbar (Wbar + jitter)]
    W = [Wbar -Wbar; -Wbar Wbar]
    logpdf = msg_out.dist.params[:log_pdf]
    msg_fwd = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_mean, dist_prec)
    v_mean_inv = cholinv(v_mean)
    l(z) = @views -0.5 * z'*W*z - 0.5 * (z[d+1:end] - m_mean)' * v_mean_inv * (z[d+1:end] - m_mean) + logpdf(z[1:d])
    # Expansion point
    point1 = unsafeMean(msg_fwd.dist * msg_out.dist)
    try
        m_joint, v_joint = NewtonMethod(l, [ point1; m_mean ])
        return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=m_joint,v=v_joint)
    catch e
        # Use cubature as a fallback
        cubature  = msg_out.dist.params[:cubature]
        mean, cov = approximate_meancov(cubature, (x) -> exp(logpdf(x)), msg_fwd.dist)
        approx_message = Message(Univariate, GaussianMeanVariance, m = mean, v = cov)
        println(approx_message)
        println(msg_mean)
        return ruleMGaussianMeanPrecisionGGD(approx_message, msg_mean, dist_prec)
    end
end

function gradientOptimization(log_joint::Function, d_log_joint::Function, m_initial, step_size)
    dim_tot = length(m_initial)
    m_total = zeros(dim_tot)
    m_average = zeros(dim_tot)
    m_new = zeros(dim_tot)
    m_old = m_initial
    satisfied = false
    step_count = 0
    while !satisfied
        m_new = m_old .+ step_size.*d_log_joint(m_old)
        if log_joint(m_new) > log_joint(m_old)
            proposal_step_size = 10*step_size
            m_proposal = m_old .+ proposal_step_size.*d_log_joint(m_old)
            if log_joint(m_proposal) > log_joint(m_new)
                m_new = m_proposal
                step_size = proposal_step_size
            end
        else
            step_size = 0.1*step_size
            m_new = m_old .+ step_size.*d_log_joint(m_old)
        end
        step_count += 1
        m_total .+= m_old
        m_average = m_total ./ step_count
        if step_count > 10
            if sum(sqrt.(((m_new.-m_average)./m_average).^2)) < dim_tot*0.001
                satisfied = true
            end
        end
        if step_count > dim_tot*250
            satisfied = true
        end
        m_old = m_new
    end
    return m_new
end

function NewtonMethod(g::Function, x_0::Array{Float64})
    dim = length(x_0)
    grad_g = (x) -> ForwardDiff.gradient(g, x)
    mode   = gradientOptimization(g, grad_g, x_0, 0.01)
    cov  = cholinv(-ForwardDiff.hessian(g, mode))
    return mode, cov
end
function collectStructuredVariationalNodeInbounds(node::GaussianMeanPrecision, entry::ScheduleEntry)
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]
    entry_posterior_factor = posteriorFactor(entry.interface.edge)
    local_edge_to_region = localEdgeToRegion(entry.interface.node)

    encountered_posterior_factors = Union{PosteriorFactor, Edge}[] # Keep track of encountered posterior factors
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_posterior_factor = posteriorFactor(node_interface.edge)

        if node_interface === entry.interface
            if (entry.message_update_rule == SVBGaussianMeanPrecisionMFND)
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            else
                push!(inbounds, nothing)
            end
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        elseif current_posterior_factor === entry_posterior_factor
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        elseif !(current_posterior_factor in encountered_posterior_factors)
            # Collect marginal from marginal dictionary (if marginal is not already accepted)
            target = local_edge_to_region[node_interface.edge]
            push!(inbounds, target_to_marginal_entry[target])
        end

        push!(encountered_posterior_factors, current_posterior_factor)
    end

    return inbounds
end
