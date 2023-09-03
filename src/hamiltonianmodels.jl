using OMEinsum

export Heisenberg, TFIsing, XXZ, J1J2, J1xJ1yJ2, Kitaev
export hamiltonian, HamiltonianModel, ExFd

function const_Sx(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sx = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if abs(i-j) == 1
            Sx[i,j] = 1/2 * sqrt(S*(S+1)-ms[i]*ms[j]) 
        end
    end
    return Sx
end

function const_Sy(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sy = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if i-j == 1
            Sy[i,j] = -1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j]) 
        elseif j-i == 1
            Sy[i,j] =  1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j]) 
        end
    end
    return Sy
end

function const_Sz(S::Real)
    dims = Int(2*S + 1)
    ms = [S-i+1 for i in 1:dims]
    Sz = zeros(ComplexF64, dims, dims)
    for i in 1:dims
        Sz[i,i] = ms[i]
    end
    return Sz
end

abstract type HamiltonianModel end

"""
    ExFd(model::HamiltonianModel, F::Real)

return a struct representing the `model` with `F` as the external field.
"""
function ExFd(model::HamiltonianModel, F)
    struct_name = string(typeof(model).name.name)*"F"

    struct_definition = """
        struct $(struct_name){T<:Real} <: HamiltonianModel
            S::T
            W::Int
            $(join([string(field) * "::T" for field in fieldnames(typeof(model))[3:end]], "\n")) 
            F::T 
        end
    """

    eval(Meta.parse(struct_definition))
    eval(Meta.parse("export $struct_name"))

    MPO2x2_definition = """
        function MPO_2x2(model::$struct_name)
            M = MPO_2x2($model)
            S = getfield(model, :S)
            F = getfield(model, :F)
            Sz = const_Sz(S)
            ISz = sum(I_S(Sz))
            M[end,:,1,:] .+= F * ISz
            return M
        end
    """
    eval(Meta.parse(MPO2x2_definition))

    return eval(Meta.parse("$struct_name($(join([string(getfield(model, field)) for field in fieldnames(typeof(model))], ", ") * ", $F"))"))
end

"""
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"""
function hamiltonian end

"""
    Heisenberg(Jz::T,Jx::T,Jy::T) where {T<:Real}

return a struct representing the spin-`S` heisenberg model with magnetisation fields
`Jz`, `Jx` and `Jy`, N is the `N-th` nearest neighbour interaction
"""
struct Heisenberg{T<:Real} <: HamiltonianModel
     S::T
     W::Int
    Jx::T
    Jy::T
    Jz::T
end
Heisenberg() = Heisenberg(1/2, 1, 1.0, 1.0, 1.0)
Heisenberg(S, W) = Heisenberg(S, W, 1.0, 1.0, 1.0)

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    S, Jx, Jy, Jz = model.S, model.Jx, model.Jy, model.Jz
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    Jx * ein"ij,kl -> ijkl"(Sx, Sx) +
    Jy * ein"ij,kl -> ijkl"(Sy, Sy) +
    Jz * ein"ij,kl -> ijkl"(Sz, Sz)
end

"""
    TFIsing(λ::Real)
return a struct representing the spin-`S` transverse field ising model with magnetisation `λ`.
"""
struct TFIsing{T<:Real} <: HamiltonianModel
    S::T
    W::Int
    λ::T
end
TFIsing(S, λ) = TFIsing(S, 1, λ)

"""
    hamiltonian(model::TFIsing)
return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::TFIsing)
    S, λ = model.S, model.λ
    Sx, Sz = 2*const_Sx(1/2), 2*const_Sz(1/2)
    D = size(Sx,1)
       - ein"ij,kl -> ijkl"(Sz,Sz) -
    λ/2 * ein"ij,kl -> ijkl"(Sx, I(D)) -
    λ/2 * ein"ij,kl -> ijkl"(I(D), Sx)
end

struct XXZ{T<:Real} <: HamiltonianModel
    Δ::T
end

"""
    hamiltonian(model::TFIsing)
return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::XXZ)
    Δ = model.Δ
    # σx, σy, σz = 2*const_Sx(0.5), 2*const_Sy(0.5), 2*const_Sz(0.5)
    Sx, Sy, Sz = const_Sx(0.5), const_Sy(0.5), const_Sz(0.5)
        ein"ij,kl -> ijkl"(Sx, Sx) +
        ein"ij,kl -> ijkl"(Sy, Sy) +
    Δ * ein"ij,kl -> ijkl"(Sz, Sz)
end

"""
    J1J2(S::T, W::Int, J1::T, J2::T) where {T<:Real}

return a struct representing the spin-`S` J1J2 model with `W`-width
"""
struct J1J2{T<:Real} <: HamiltonianModel
     S::T
     W::Int
    J1::T
    J2::T
end
J1J2(W, J2) = J1J2(1/2, W, 1.0, J2)

"""
    J1xJ1yJ2(S::T, W::Int, Jx::T, Jy::T, J2::T) where {T<:Real}

return a struct representing the spin-`S` J1xJ1yJ2 model with `W`-width
"""
struct J1xJ1yJ2{T<:Real} <: HamiltonianModel
     S::T
     W::Int
    J1x::T
    J1y::T
    J2::T
end
J1xJ1yJ2(W, J1y, J2) = J1xJ1yJ2(1/2, W, J1x, J1y, J2)

"""
    Kitaev(S::T, W::Int) where {T<:Real}

return a struct representing the spin-`S` Kitaev model with `W`-width
"""
struct Kitaev{T<:Real} <: HamiltonianModel
     S::T
     W::Int
end
Kitaev(W) = Kitaev(1/2, W)