using OMEinsum

export Heisenberg
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
    for j in 1:dims, i in 1:dims
        if i-j == 0
            Sz[i,j] = ms[i]
        end
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

return a struct representing the Spin-`S` heisenberg model with magnetisation fields
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