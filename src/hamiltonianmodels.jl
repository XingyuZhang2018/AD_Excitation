using OMEinsum

export Heisenberg, TFIsing, XXZ
export hamiltonian, HamiltonianModel

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
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"""
function hamiltonian end

"""
    Heisenberg(Jz::T,Jx::T,Jy::T) where {T<:Real}

return a struct representing the spin-`S` heisenberg model with magnetisation fields
`Jz`, `Jx` and `Jy`
"""
struct Heisenberg{T<:Real} <: HamiltonianModel
     S::T
    Jx::T
    Jy::T
    Jz::T
end
Heisenberg() = Heisenberg(1/2, 1.0, 1.0, 1.0)
Heisenberg(S) = Heisenberg(S, 1.0, 1.0, 1.0)

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    S, Jx, Jy, Jz = model.S, model.Jx, model.Jy, model.Jz
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    Jx * ein"ij,kl -> ijkl"(Sx, Sx) -
    Jy * ein"ij,kl -> ijkl"(Sy, Sy) -
    Jz * ein"ij,kl -> ijkl"(Sz, Sz)
end

"""
    TFIsing(hx::Real)
return a struct representing the spin-`S` transverse field ising model with magnetisation `hx`.
"""
struct TFIsing{T<:Real} <: HamiltonianModel
     S::T
    hx::T
end

"""
    hamiltonian(model::TFIsing)
return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::TFIsing)
    S, hx = model.S, model.hx
    Sx, Sz = const_Sx(S), const_Sz(S)
    D = size(Sx,1)
       - ein"ij,kl -> ijkl"(Sz,Sz) -
    hx/2 * ein"ij,kl -> ijkl"(Sx, I(D)) -
    hx/2 * ein"ij,kl -> ijkl"(I(D), Sx)
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
        ein"ij,kl -> ijkl"(Sx, Sx) -
        ein"ij,kl -> ijkl"(Sy, Sy) -
    Δ * ein"ij,kl -> ijkl"(Sz, Sz)
end