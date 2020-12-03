# const sigma_points = [-5.387480890011233, -4.603682449550744, -3.9447640401156265, -3.3478545673832163, -2.7888060584281296, -2.2549740020892757, -1.7385377121165857, -1.2340762153953255, -0.7374737285453978, -0.24534070830090382, 0.24534070830090382, 0.7374737285453978, 1.2340762153953255, 1.7385377121165857, 2.2549740020892757, 2.7888060584281296, 3.3478545673832163, 3.9447640401156265, 4.603682449550744, 5.387480890011233]
# const sigma_weights = [2.2293936455341583e-13, 4.399340992273223e-10, 1.0860693707692783e-7, 7.802556478532184e-6, 0.00022833863601635774, 0.0032437733422378905, 0.024810520887463966, 0.10901720602002457, 0.28667550536283243, 0.4622436696006102, 0.4622436696006102, 0.28667550536283243, 0.10901720602002457, 0.024810520887463966, 0.0032437733422378905, 0.00022833863601635774, 7.802556478532184e-6, 1.0860693707692783e-7, 4.399340992273223e-10, 2.2293936455341583e-13]

import LinearAlgebra: mul!, axpy!
#
# const sigma_points = [-2.0201828704560856, -0.9585724646138196, -8.881784197001252e-16, 0.9585724646138196, 2.0201828704560856]
# const sigma_weights = [0.019953242059045872, 0.3936193231522407, 0.9453087204829428, 0.3936193231522407, 0.019953242059045872]

const product = Iterators.product
const repeated = Iterators.repeated

const sigma_points = [-13.406487338144908, -12.823799749487808, -12.342964222859672, -11.915061943114164, -11.52141540078703, -11.152404385585125, -10.802260753684713, -10.467185421342812, -10.144509941292846, -9.832269807777967, -9.528965823390115, -9.233420890219161, -8.944689217325474, -8.661996168134518, -8.384696940416266, -8.112247311162792, -7.84418238446082, -7.580100807857489, -7.319652822304534, -7.062531060248865, -6.808463352858795, -6.557207031921539, -6.308544361112134, -6.062278832614302, -5.818232135203517, -5.576241649329924, -5.33615836013836, -5.097845105089136, -4.8611750917912095, -4.6260306357871555, -4.392302078682683, -4.15988685513103, -3.9286886834276706, -3.6986168593184914, -3.469585636418589, -3.241513679631013, -3.014323580331155, -2.787941423981989, -2.5622964023726076, -2.3373204639068783, -2.112947996371188, -1.8891155374270083, -1.6657615087415094, -1.4428259702159327, -1.2202503912189528, -0.9979774360981053, -0.7759507615401456, -0.5541148235916169, -0.3324146923422318, -0.11079587242243949, 0.11079587242243949, 0.3324146923422318, 0.5541148235916169, 0.7759507615401456, 0.9979774360981053, 1.2202503912189528, 1.4428259702159327, 1.6657615087415094, 1.8891155374270083, 2.112947996371188, 2.3373204639068783, 2.5622964023726076, 2.787941423981989, 3.014323580331155, 3.241513679631013, 3.469585636418589, 3.6986168593184914, 3.9286886834276706, 4.15988685513103, 4.392302078682683, 4.6260306357871555, 4.8611750917912095, 5.097845105089136, 5.33615836013836, 5.576241649329924, 5.818232135203517, 6.062278832614302, 6.308544361112134, 6.557207031921539, 6.808463352858795, 7.062531060248865, 7.319652822304534, 7.580100807857489, 7.84418238446082, 8.112247311162792, 8.384696940416266, 8.661996168134518, 8.944689217325474, 9.233420890219161, 9.528965823390115, 9.832269807777967, 10.144509941292846, 10.467185421342812, 10.802260753684713, 11.152404385585125, 11.52141540078703, 11.915061943114164, 12.342964222859672, 12.823799749487808, 13.406487338144908]
#
const sigma_weights = [5.90806786475396e-79, 1.972860574879216e-72, 3.083028990003297e-67, 9.019222303693804e-63, 8.518883081761774e-59, 3.459477936475577e-55, 7.191529463463525e-52, 8.597563954825022e-49, 6.4207252053483165e-46, 3.1852178778359564e-43, 1.1004706827141981e-40, 2.7487848843571714e-38, 5.1162326043853164e-36, 7.274572596887586e-34, 8.067434278709346e-32, 7.101812226384877e-30, 5.037791166213212e-28, 2.917350072629348e-26, 1.3948415260687509e-24, 5.561026961659241e-23, 1.864997675130272e-21, 5.302316183131963e-20, 1.28683292112117e-18, 2.6824921647603466e-17, 4.829835321703033e-16, 7.548896877915255e-15, 1.0288749373509815e-13, 1.2278785144101149e-12, 1.2879038257315609e-11, 1.1913006349290596e-10, 9.747921253871486e-10, 7.075857283889495e-9, 4.5681275084849026e-8, 2.6290974837537006e-7, 1.3517971591103645e-6, 6.221524817777747e-6, 2.5676159384548995e-5, 9.517162778551009e-5, 0.00031729197104329556, 0.0009526921885486135, 0.002579273260059073, 0.006303000285608099, 0.013915665220231849, 0.027779127385933522, 0.05017581267742825, 0.08205182739122392, 0.12153798684410465, 0.16313003050278302, 0.1984628502541864, 0.21889262958743966, 0.21889262958743966, 0.1984628502541864, 0.16313003050278302, 0.12153798684410465, 0.08205182739122392, 0.05017581267742825, 0.027779127385933522, 0.013915665220231849, 0.006303000285608099, 0.002579273260059073, 0.0009526921885486135, 0.00031729197104329556, 9.517162778551009e-5, 2.5676159384548995e-5, 6.221524817777747e-6, 1.3517971591103645e-6, 2.6290974837537006e-7, 4.5681275084849026e-8, 7.075857283889495e-9, 9.747921253871486e-10, 1.1913006349290596e-10, 1.2879038257315609e-11, 1.2278785144101149e-12, 1.0288749373509815e-13, 7.548896877915255e-15, 4.829835321703033e-16, 2.6824921647603466e-17, 1.28683292112117e-18, 5.302316183131963e-20, 1.864997675130272e-21, 5.561026961659241e-23, 1.3948415260687509e-24, 2.917350072629348e-26, 5.037791166213212e-28, 7.101812226384877e-30, 8.067434278709346e-32, 7.274572596887586e-34, 5.1162326043853164e-36, 2.7487848843571714e-38, 1.1004706827141981e-40, 3.1852178778359564e-43, 6.4207252053483165e-46, 8.597563954825022e-49, 7.191529463463525e-52, 3.459477936475577e-55, 8.518883081761774e-59, 9.019222303693804e-63, 3.083028990003297e-67, 1.972860574879216e-72, 5.90806786475396e-79]

function multiDimensionalPointsWeightsGrid(ndims::Int, p::Int)
    # we use fixed p = 20 for the beginning

    # creates lazy multi-dimensional grid
    witer = product(repeated(sigma_weights, ndims)...)
    piter = product(repeated(sigma_points, ndims)...)

    return witer, piter
end

struct GaussHermiteCubature{N, WI, PI}
    ndims :: Int
    p     :: Int
    witer :: WI
    piter :: PI
end

function ghcubature(ndims::Int, p::Int)
    witer, piter = multiDimensionalPointsWeightsGrid(ndims, p)
    return GaussHermiteCubature{ndims, typeof(witer), typeof(piter)}(ndims, p, witer, piter)
end

function getweights(cubature::GaussHermiteCubature)
    return Base.Generator(cubature.witer) do pweight
        return prod(pweight)
    end
end

function getpoints(cubature::GaussHermiteCubature, m, P)
    sqrtP = sqrt(P)
    sqrt2 = sqrt(2)

    tbuffer = similar(m)
    pbuffer = similar(m)
    return Base.Generator(cubature.piter) do ptuple
        copyto!(pbuffer, ptuple)
        copyto!(tbuffer, m)
        return mul!(tbuffer, sqrtP, pbuffer, sqrt2, 1.0) # point = m + sqrt2 * sqrtP * p
    end
end

function approximate_meancov(cubature::GaussHermiteCubature, g, distribution)
    ndims = cubature.ndims
    m, P = ForneyLab.unsafeMeanCov(distribution)

    weights = getweights(cubature)
    points  = getpoints(cubature, m, P)

    cs = similar(m, eltype(m), length(weights))
    norm = 0.0
    mean = zeros(ndims)

    sqrtpi = (pi ^ (ndims / 2))

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        # mean = mean + point * weight * g(point)
        broadcast!(*, point, point, cv)  # point *= cv
        broadcast!(+, mean, mean, point) # mean += point
        norm += cv

        @inbounds cs[index] = cv
    end

    norm /= sqrtpi

    broadcast!(/, mean, mean, norm)
    broadcast!(/, mean, mean, sqrtpi)

    cov = zeros(ndims, ndims)
    foreach(enumerate(zip(points, cs))) do (index, (point, c))
        broadcast!(-, point, point, mean)                # point -= mean
        mul!(cov, point, reshape(point, (1, ndims)), c, 1.0) # cov = cov + c * (point)⋅(point)' where c = weight * g(point)
    end

    broadcast!(/, cov, cov, norm)
    broadcast!(/, cov, cov, sqrtpi)

    return mean, cov
end

function approximate_norm(cubature::GaussHermiteCubature, g, distribution)
    ndims = cubature.ndims
    m, P = ForneyLab.unsafeMeanCov(distribution)

    weights = getweights(cubature)
    points  = getpoints(cubature, m, P)

    norm = 0.0

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv
        norm += cv
    end

    norm /= (pi ^ (ndims / 2))

    return norm
end

function approximate_kernel_expectation(cubature::GaussHermiteCubature, g, distribution)
    ndims = cubature.ndims
    m, P = ForneyLab.unsafeMeanCov(distribution)

    weights = getweights(cubature)
    points  = getpoints(cubature, m, P)

    gbar = zeros(ndims, ndims)
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end

    sqrtpi = (pi ^ (ndims / 2))
    broadcast!(/, gbar, gbar, sqrtpi)
    return gbar
end

function getweights(gh::GaussHermiteCubature{1})
    return Base.Generator(gh.witer) do weight
        return weight[1]
    end
end

function getpoints(gh::GaussHermiteCubature{1}, mean::T, variance::T) where { T <: Real }
    sqrt2V = sqrt(2 * variance)
    return Base.Generator(gh.piter) do point
        return mean + sqrt2V * point[1]
    end
end

function approximate_meancov(gh::GaussHermiteCubature{1}, g::Function, distribution)
    m, v = ForneyLab.unsafeMeanCov(distribution)

    weights = getweights(gh)
    points  = getpoints(gh, m, v)

    cs   = Vector{eltype(m)}(undef, length(weights))
    norm = 0.0
    mean = 0.0

    sqrtpi = sqrt(pi)

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        mean += point * cv
        norm += cv

        @inbounds cs[index] = cv
    end

    norm /= sqrtpi

    mean /= norm
    mean /= sqrtpi

    var = 0.0
    for (index, (point, c)) in enumerate(zip(points, cs))
        point -= mean
        var += c * point ^ 2
    end

    var /= norm
    var /= sqrtpi

    return mean, var
end
