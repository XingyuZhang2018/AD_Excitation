export MPO

"""
    MPO(model<:HamiltonianModel)

return the MPO of the `model` as a four-bond tensor.
"""
function MPO end

"""
    MPO(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function MPO(model::Heisenberg)
    S, N, Jx, Jy, Jz = model.S, model.N, model.Jx, model.Jy, model.Jz
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    d = size(Sx, 1)
    if N == 1
        M = zeros(ComplexF64, 5, d, 5, d)
        M[1,:,1,:] .= I(d)
        M[2,:,1,:] .= Jx * Sx
        M[3,:,1,:] .= Jy * Sy
        M[4,:,1,:] .= Jz * Sz
        M[5,:,2,:] .= Sx
        M[5,:,3,:] .= Sy
        M[5,:,4,:] .= Sz
        M[5,:,5,:] .= I(d)
    else
        M = zeros(ComplexF64, 5+3*N, d, 5+3*N, d)
        M[2,:,1,:] .= Jx * Sx
        M[2+N,:,1,:] .= Jy * Sy
        M[2+2*N,:,1,:] .= Jz * Sz
        M[2+3*N,:,1,:] .= Jx * Sx
        M[3+3*N,:,1,:] .= Jy * Sy
        M[4+3*N,:,1,:] .= Jz * Sz
        M[5+3*N,:,1+N,:] .= Sx
        M[5+3*N,:,1+2*N,:] .= Sy
        M[5+3*N,:,1+3*N,:] .= Sz
        M[5+3*N,:,2+3*N,:] .= Sx
        M[5+3*N,:,3+3*N,:] .= Sy
        M[5+3*N,:,4+3*N,:] .= Sz
        for i in 2:N
            M[i+1,:,i,:] .= I(d)
            M[i+1+N,:,i+N,:] .= I(d)
            M[i+1+2*N,:,i+2*N,:] .= I(d)
        end
        M[1,:,1,:] .= I(d)
        M[5+3*N,:,5+3*N,:] .= I(d)
    end
    return M
end

"""
    MPO(model::TFising)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function MPO(model::TFIsing)
    S, hx = model.S, model.hx
    σx, σz = 2*const_Sx(1/2), 2*const_Sz(1/2)
    d = size(σx, 1)
    M = zeros(ComplexF64, 3, d, 3, d)
    M[1,:,1,:] .= I(d)
    M[2,:,1,:] .= -σz
    M[3,:,1,:] .= -hx * σx
    M[3,:,2,:] .= σz
    M[3,:,3,:] .= I(d)
    return M
end