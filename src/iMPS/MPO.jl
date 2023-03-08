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
    S, W, Jx, Jy, Jz = model.S, model.W, model.Jx, model.Jy, model.Jz
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    d = size(Sx, 1)
    if W == 1
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
        M = zeros(ComplexF64, 2+3*W, d, 2+3*W, d)
        M[2,:,1,:] .= Jx * Sx
        M[2+W,:,1,:] .= Jy * Sy
        M[2+2*W,:,1,:] .= Jz * Sz
        M[2+3*W,:,1+W,:] .= Sx
        M[2+3*W,:,1+2*W,:] .= Sy
        M[2+3*W,:,1+3*W,:] .= Sz
        M[2+3*W,:,2,:] .= Sx
        M[2+3*W,:,2+W,:] .= Sy
        M[2+3*W,:,2+2*W,:] .= Sz
        for i in 2:W
            M[i+1,:,i,:] .= I(d)
            M[i+1+W,:,i+W,:] .= I(d)
            M[i+1+2*W,:,i+2*W,:] .= I(d)
        end
        M[1,:,1,:] .= I(d)
        M[2+3*W,:,2+3*W,:] .= I(d)
    end
    return M
end

"""
    MPO(model::TFising)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function MPO(model::TFIsing)
    S, W, λ = model.S, model.W, model.λ
    Sx, Sz = 2*const_Sx(S), 2*const_Sz(S)
    d = size(Sx, 1)
    if W == 1
        M = zeros(ComplexF64, 3, d, 3, d)
        M[1,:,1,:] .= I(d)
        M[2,:,1,:] .= -Sz
        M[3,:,1,:] .= -λ * Sx
        M[3,:,2,:] .= Sz
        M[3,:,3,:] .= I(d)
    else
        M = zeros(ComplexF64, 2+W, d, 2+W, d)
        M[2,:,1,:] .= -Sz
        M[2+W,:,1+W,:] .= Sz
        M[2+W,:,2,:] .= Sz
        for i in 2:W
            M[i+1,:,i,:] .= I(d)
        end
        M[1,:,1,:] .= I(d)
        M[2+W,:,2+W,:] .= I(d)
        M[2+W,:,1,:] .= -λ * Sx
    end
    return M
end

"""
    MPO(model::J1J2)

return the J1-J2 Heisenberg MPO with width `W` in helix structure.
"""
function MPO(model::J1J2)
    S, W, J1, J2 = model.S, model.W, model.J1, model.J2
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    d = size(Sx, 1)
    M = zeros(ComplexF64, 5+3*W, d, 5+3*W, d)
    M[2,    :,1,:] .= Sx
    M[3+W,  :,1,:] .= Sy
    M[4+2*W,:,1,:] .= Sz

    for i in 2:W+1
        M[i+1,    :,i,      :] .= I(d)
        M[i+2+W,  :,i+1+W,  :] .= I(d)
        M[i+3+2*W,:,i+2+2*W,:] .= I(d)
    end

    M[1,    :,1,    :] .= I(d)
    M[5+3*W,:,5+3*W,:] .= I(d)
    
    # nearest neighbor
    M[5+3*W,:,2,    :] .= J1 * Sx
    M[5+3*W,:,3+W,  :] .= J1 * Sy
    M[5+3*W,:,4+2*W,:] .= J1 * Sz

    # W+1 nearest neighbor
    M[5+3*W,:,2+W,  :] .= J2 * Sx
    M[5+3*W,:,3+2*W,:] .= J2 * Sy
    M[5+3*W,:,4+3*W,:] .= J2 * Sz

    # W  nearest neighbor
    M[5+3*W,:,1+W,  :] .= J1 * Sx
    M[5+3*W,:,2+2*W,:] .= J1 * Sy
    M[5+3*W,:,3+3*W,:] .= J1 * Sz

    # W-1  nearest neighbor
    M[5+3*W,:,W,    :] .= J2 * Sx
    M[5+3*W,:,1+2*W,:] .= J2 * Sy
    M[5+3*W,:,2+3*W,:] .= J2 * Sz
    return M
end
