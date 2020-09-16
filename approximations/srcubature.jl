
import LinearAlgebra: mul!

struct SphericalRadialCubature
    ndims :: Int
end

srcubature(ndims::Int) = SphericalRadialCubature(ndims)

function getweights(cubature::SphericalRadialCubature)
    d = cubature.ndims
    return Base.Generator(1:2d) do _
        return 1.0 / (2.0 * d)
    end
end

function getpoints(cubature::SphericalRadialCubature, m, P)
    d = cubature.ndims

    if isa(P, Diagonal)
        L = sqrt(P) # Matrix square root
    else
        L = sqrt(Hermitian(P))
    end

    tmpbuffer = zeros(d)
    sigma_points = Base.Generator(1:2d) do i
        tmpbuffer[rem((i - 1), d) + 1] = sqrt(d) * (-1)^(div(i - 1, d))
        if i !== 1
            tmpbuffer[rem((i - 2), d) + 1] = 0.0
        end
        return tmpbuffer
    end

    tbuffer = similar(m)
    return Base.Generator(sigma_points) do point
        copyto!(tbuffer, m)
        return mul!(tbuffer, L, point, 1.0, 1.0) # point = m + 1.0 * L * point
    end
end

function approximate_meancov(cubature::SphericalRadialCubature, g, distribution)
    ndims = cubature.ndims

    c    = spherical_expectations(cubature, g, distribution)
    mean = spherical_expectations(cubature, (s) -> g(s) * s / c, distribution)
    cov  = spherical_expectations(cubature, (s) -> g(s) * (s - mean) * (s - mean)' / c, distribution)

    return mean, cov
end

function spherical_expectations(cubature::SphericalRadialCubature, g, distribution)
    m, V = ForneyLab.unsafeMeanCov(distribution)

    weights = getweights(cubature)
    points  = getpoints(cubature, m, V)

    gs = Base.Generator(points) do point
        return g(point)
    end

    return mapreduce(t -> t[1] * t[2], +, zip(weights, gs))
end
