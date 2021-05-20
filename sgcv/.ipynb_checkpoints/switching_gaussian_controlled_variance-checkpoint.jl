import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov, unsafeWeightedMeanPrecision

export SwitchingGaussianControlledVariance, averageEnergy, slug

mutable struct SwitchingGaussianControlledVariance <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function SwitchingGaussianControlledVariance(y, x, z, κ, ω, s; id=generateId(SwitchingGaussianControlledVariance))
        @ensureVariables(y, x, z, κ, ω, s)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)
        self.i[:κ] = self.interfaces[4] = associate!(Interface(self), κ)
        self.i[:ω] = self.interfaces[5] = associate!(Interface(self), ω)
        self.i[:s] = self.interfaces[6] = associate!(Interface(self), s)

        return self
    end
end

slug(::Type{SwitchingGaussianControlledVariance}) = "SGCV"

function ψ(yx)
    m, V = unsafeMeanCov(yx)
    (m[1] - m[2])*(m[1] - m[2]) + V[1] + V[4] - V[3] - V[2]
end

function ϕ(z, κ, ω, s)
    ms = unsafeMean(s)
    mω, Vω = unsafeMeanCov(ω) 
    mz, vz = unsafeMeanCov(z)
    mκ, Vκ = unsafeMeanCov(κ)
    exp(-ms'*mκ*mz - ms'*mω + 0.5((ms'*mκ)^2*vz + mz^2*ms'*Vκ*ms + ms'*Vκ*ms*vz + ms'*Vω*ms))
end

# Average energy functional
function averageEnergy(::Type{SwitchingGaussianControlledVariance}, marg_y_x::ProbabilityDistribution{Multivariate}, marg_z::ProbabilityDistribution{Univariate}, 
                        marg_κ::ProbabilityDistribution{Multivariate}, marg_ω::ProbabilityDistribution{Multivariate}, marg_s::ProbabilityDistribution{Univariate})
    m_z, var_z = unsafeMeanCov(marg_z)
    m_κ, var_κ = unsafeMeanCov(marg_κ)
    m_ω, var_ω = unsafeMeanCov(marg_ω)
    m_s = unsafeMean(marg_s)

    0.5log(2*pi) + 0.5*(m_s'*m_κ*m_z + m_s'*m_ω) + 0.5*(ψ(marg_y_x)*ϕ(marg_z, marg_κ, marg_ω, marg_s))
end