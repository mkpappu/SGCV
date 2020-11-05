import ForneyLab: inferUpdateRule!, collectInboundTypes, leaftypes, isApplicable, prod!,
                  @symmetrical, SampleList, VariateType, SPEqualityFnG, unsafeMean, Weights,
                  inferMarginalRule, Cluster
import GCV: SVBGaussianMeanPrecisionMEND, MGaussianMeanPrecisionEGD
export inferUpdateRule!, prod!, inferMarginalRule

function inferUpdateRule!(entry::ScheduleEntry,
                          rule_type::Type{T},
                          inferred_outbound_types::Dict{Interface, <:Type}
                         ) where T<:SumProductRule
    # Collect inbound types
    inbound_types = collectInboundTypes(entry, rule_type, inferred_outbound_types)

    # Find applicable rule(s)
    applicable_rules = Type[]
    for rule in leaftypes(entry.message_update_rule)
        if isApplicable(rule, inbound_types)
            push!(applicable_rules, rule)
        end
    end

    outbounds = collect(keys(inferred_outbound_types))
    for i in 1:length(outbounds)
        if occursin(string(:gaussiancontrolledvariance_), string(outbounds[i].node.id)) && length(applicable_rules) > 1
            filter!(e -> e≠SVBGaussianMeanPrecisionMEND, applicable_rules)
            filter!(e -> e≠SPEqualityFnG, applicable_rules)
            break
        end
    end

    # Select and set applicable rule
    if isempty(applicable_rules)
        if isa(entry.interface.node, CompositeFactor)
            # No 'shortcut rule' available for CompositeFactor.
            # Try to fall back to msg passing on the internal graph.
            entry.internal_schedule = internalSumProductSchedule(entry.interface.node, entry.interface, inferred_outbound_types)
            entry.message_update_rule = entry.internal_schedule[end].message_update_rule
        else
            error("No applicable $(rule_type) update for $(typeof(entry.interface.node)) node with inbound types: $(join(inbound_types, ", "))")
        end
    elseif length(applicable_rules) > 1
        error("Multiple applicable $(rule_type) updates for $(typeof(entry.interface.node)) node with inbound types: $(join(inbound_types, ", "))")
    else
        entry.message_update_rule = first(applicable_rules)
    end

    return entry
end

function inferUpdateRule!(entry::ScheduleEntry,
                          rule_type::Type{T},
                          inferred_outbound_types::Dict{Interface, Type}
                         ) where T<:StructuredVariationalRule
    # Collect inbound types
    inbound_types = collectInboundTypes(entry, rule_type, inferred_outbound_types)

    # Find applicable rule(s)
    applicable_rules = Type[]
    for rule in leaftypes(entry.message_update_rule)
        if isApplicable(rule, inbound_types)
            push!(applicable_rules, rule)
        end
    end

    outbounds = collect(keys(inferred_outbound_types))
    for i in 1:length(outbounds)
        if occursin(string(:switchinggaussiancontrolledvariance_), string(outbounds[i].node.id)) && length(applicable_rules) > 1
            filter!(e -> e≠SPEqualityFnG, applicable_rules)
            filter!(e -> e≠SVBGaussianMeanPrecisionMEND, applicable_rules)
            break
        elseif occursin(string(:gaussiancontrolledvariance_), string(outbounds[i].node.id)) && length(applicable_rules) > 1
            filter!(e -> e≠SPEqualityFnG, applicable_rules)
            filter!(e -> e≠SVBGaussianMeanPrecisionMEND, applicable_rules)
            break
        end

    end

    # Select and set applicable rule
    if isempty(applicable_rules)
        error("No applicable $(rule_type) update for $(typeof(entry.interface.node)) node with inbound types: $(join(inbound_types, ", "))")
    elseif length(applicable_rules) > 1
        println(applicable_rules)
        error("Multiple applicable $(rule_type) updates for $(typeof(entry.interface.node)) node with inbound types: $(join(inbound_types, ", "))")
    else
        entry.message_update_rule = first(applicable_rules)
    end

    return entry
end

"""
Infer the rule that computes the joint marginal over `cluster`
"""
function inferMarginalRule(cluster::Cluster, inbound_types::Vector{<:Type})
    # Find applicable rule(s)
    applicable_rules = Type[]
    for rule in leaftypes(MarginalRule{typeof(cluster.node)})
        if isApplicable(rule, inbound_types)
            push!(applicable_rules, rule)
        end
    end

    if MGaussianMeanPrecisionEGD in applicable_rules && length(applicable_rules) > 1
        filter!(e -> e≠MGaussianMeanPrecisionEGD, applicable_rules)
    end

    # Select and set applicable rule
    if isempty(applicable_rules)
        error("No applicable marginal update rule for $(typeof(cluster.node)) node with inbound types: $(join(inbound_types, ", "))")
    elseif length(applicable_rules) > 1
        error("Multiple applicable marginal update rules for $(typeof(cluster.node)) node with inbound types: $(join(inbound_types, ", "))")
    else
        marginal_update_rule = first(applicable_rules)
    end

    return marginal_update_rule
end

@symmetrical function prod!(
    x::ProbabilityDistribution{V}, # Includes function distributions
    y::ProbabilityDistribution{V, SampleList},
    z::ProbabilityDistribution{V, SampleList}=ProbabilityDistribution(V, SampleList, s=[0.0], w=[1.0])) where V<:VariateType

    samples = y.params[:s]
    n_samples = length(samples)
    log_samples_x = logPdf.([x], samples)

    # Compute sample weights
    w_raw_x = exp.(log_samples_x)
    w_prod = w_raw_x.*y.params[:w]
    weights = w_prod./sum(w_prod) # Normalize weights

    # Resample if required
    n_eff = 1/sum(weights.^2) # Effective number of particles
    if n_eff < n_samples/10
        samples = sample(samples, Weights(weights), n_samples) # Use StatsBase for resampling
        weights = ones(n_samples)./n_samples
    end

    # TODO: no entropy is computed here; include computation?
    z.params[:w] = weights
    z.params[:s] = samples

    m = unsafeMean(z)

    if typeof(samples) == Array{SparseVector{Float64,Int64},1}
        p = ProbabilityDistribution(Univariate, Categorical, p=m)
        z.params[:entropy] = differentialEntropy(p)
    end

    return z
end
