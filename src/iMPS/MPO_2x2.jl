export MPO_2x2

"""
    MPO_2x2(model<:HamiltonianModel)

site label
```
    │  │  │  │ 
   ─3──4──3──4─
    │  │  │  │     
   ─1──2──1──2─
    │  │  │  │ 
   ─3──4──3──4─
    │  │  │  │     
   ─1──2──1──2─
    │  │  │  │
```
```
return the 2x2 helix MPO of the `model` as a four-bond tensor.
"""
function MPO_2x2 end

contract4(x) = (d = size(x[1],1); reshape(ein"((ae,bf),cg),dh->abcdefgh"(x...), (d^4,d^4)))
I_S(S) = (d = size(S, 1); Id = I(d); 
         [contract4(circshift([S, Id, Id, Id], i)) for i in 0:3]
        )
I_4(d) = (Id = I(d); contract4([Id, Id, Id, Id]))
H_on_site(S) = (d = size(S, 1); Id = I(d); 
                mapreduce(contract4, +,
                          [
                           [ S,  S, Id, Id], 
                           [ S, Id,  S, Id], 
                           [Id,  S, Id,  S], 
                           [Id, Id,  S,  S]
                          ]
                          )
               )

function MPO_2x2(model::Heisenberg)
    # some constants
    S, W, Jx, Jy, Jz = model.S, model.W, model.Jx, model.Jy, model.Jz
    @assert W >= 2 "The width of the model must be at least 2."
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    d = size(Sx, 1)
    M = zeros(ComplexF64, 2+6W, d^4, 2+6W, d^4)

    # precompute the 2x2 operators
    ISx, ISy, ISz, I4 = I_S(Sx), I_S(Sy), I_S(Sz), I_4(d)
    M[2,   :,1,:] .= Jx * ISx[4]
    M[2+W, :,1,:] .= Jx * ISx[2]
    M[1+2W,:,1,:] .= Jx * ISx[3]

    M[2+2W,:,1,:] .= Jy * ISy[4]
    M[2+3W,:,1,:] .= Jy * ISy[2]
    M[1+4W,:,1,:] .= Jy * ISy[3]

    M[2+4W,:,1,:] .= Jz * ISz[4]
    M[2+5W,:,1,:] .= Jz * ISz[2]
    M[1+6W,:,1,:] .= Jz * ISz[3]

    for i in 2:W, j in 0:5
        M[i+1+j*W,:,i+j*W ,:] .= I4
    end

    M[   1,:,   1,:] .= I4
    M[2+6W,:,2+6W,:] .= I4

    # on site
    M[2+6W,:,1,:] .= Jx * H_on_site(Sx) + Jy * H_on_site(Sy) + Jz * H_on_site(Sz)

    # nearest neighbor
    M[2+6W,:,   2,:] .= ISx[2]
    M[2+6W,:,1+2W,:] .= ISx[1]  # both for W nearest neighbor
    M[2+6W,:,2+2W,:] .= ISy[2]
    M[2+6W,:,1+4W,:] .= ISy[1]
    M[2+6W,:,2+4W,:] .= ISz[2]
    M[2+6W,:,1+6W,:] .= ISz[1]

    # W nearest neighbor
    M[2+6W,:,1+W, :] .= ISx[3]
    M[2+6W,:,1+3W,:] .= ISy[3]
    M[2+6W,:,1+5W,:] .= ISz[3]

    return M
end

H_on_J1xJ1yJ2_site(S,Jx,Jy,J2) = (d = size(S, 1); Id = I(d); 
                                mapreduce(contract4, +,
                                          [
                                           [Jx* S,  S, Id, Id], 
                                           [Jy* S, Id,  S, Id], 
                                           [Jy*Id,  S, Id,  S], 
                                           [Jx*Id, Id,  S,  S],
                                           [J2* S, Id, Id,  S],
                                           [J2*Id,  S,  S, Id]
                                          ]
                                         )
                               )

function MPO_2x2(model::J1xJ1yJ2)
    # some constants
    S, W, J1x, J1y, J2 = model.S, model.W, model.J1x, model.J1y, model.J2
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    d = size(Sx, 1)
    ISx, ISy, ISz, I4 = I_S(Sx), I_S(Sy), I_S(Sz), I_4(d)
    M = zeros(ComplexF64, 8+6W, d^4, 8+6W, d^4)

    if W == 1
        # precompute the 2x2 operators
        M[2,:,1,:] .= J2 * ISx[2] + J1y * ISx[4]
        M[3,:,1,:] .= J2 * ISx[1] + J1y * ISx[3]
        M[4,:,1,:] .= J2 * ISx[4]
        M[5,:,1,:] .= J2 * ISx[3]

        M[6,:,1,:] .= J2 * ISy[2] + J1y * ISy[4]
        M[7,:,1,:] .= J2 * ISy[1] + J1y * ISy[3]
        M[8,:,1,:] .= J2 * ISy[4]
        M[9,:,1,:] .= J2 * ISy[3]

        M[10,:,1,:] .= J2 * ISz[2] + J1y * ISz[4]
        M[11,:,1,:] .= J2 * ISz[1] + J1y * ISz[3]
        M[12,:,1,:] .= J2 * ISz[4]
        M[13,:,1,:] .= J2 * ISz[3]

        M[ 1,:, 1,:] .= I4
        M[14,:,14,:] .= I4

        # on site
        M[14,:,1,:] .= H_on_J1xJ1yJ2_site(Sx,0,J1y,0) + 
                       H_on_J1xJ1yJ2_site(Sy,0,J1y,0) + 
                       H_on_J1xJ1yJ2_site(Sz,0,J1y,0)

        # nearest neighbor
        M[14,:,2,:] .= ISx[2]
        M[14,:,3,:] .= ISx[1]
        M[14,:,4,:] .= ISx[4]
        M[14,:,5,:] .= ISx[3]

        M[14,:,6,:] .= ISy[2]
        M[14,:,7,:] .= ISy[1]
        M[14,:,8,:] .= ISy[4]
        M[14,:,9,:] .= ISy[3]

        M[14,:,10,:] .= ISz[2]
        M[14,:,11,:] .= ISz[1]
        M[14,:,12,:] .= ISz[4]
        M[14,:,13,:] .= ISz[3]
    else
        # precompute the 2x2 operators
        
        M[2,   :,1,:] .= ISx[4]
        M[3+W, :,1,:] .= ISx[2]
        M[3+2W,:,1,:] .= ISx[3]

        M[4+2W,:,1,:] .= ISy[4]
        M[5+3W,:,1,:] .= ISy[2]
        M[5+4W,:,1,:] .= ISy[3]

        M[6+4W,:,1,:] .= ISz[4]
        M[7+5W,:,1,:] .= ISz[2]
        M[7+6W,:,1,:] .= ISz[3]

        for i in 2:W+1, j in 0:2
            M[i+1+j*(2W+2),:,i+j*(2W+2),:] .= I4
        end

        for i in W+3:2W+1, j in 0:2
            M[i+1+j*(2W+2),:,i+j*(2W+2),:] .= I4
        end

        M[   1,:,   1,:] .= I4
        M[8+6W,:,8+6W,:] .= I4

        # on site
        M[8+6W,:,1,:] .= H_on_J1xJ1yJ2_site(Sx,J1x,J1y,J2) + 
                         H_on_J1xJ1yJ2_site(Sy,J1x,J1y,J2) + 
                         H_on_J1xJ1yJ2_site(Sz,J1x,J1y,J2)

        # nearest neighbor
        M[8+6W,:,   2,:] .= J2  * ISx[1] + J1y * ISx[2] #4y
        M[8+6W,:,3+2W,:] .= J1y * ISx[1] + J2  * ISx[2] #3y 
        M[8+6W,:,4+2W,:] .= J2  * ISy[1] + J1y * ISy[2] #4y
        M[8+6W,:,5+4W,:] .= J1y * ISy[1] + J2  * ISy[2] #3y
        M[8+6W,:,6+4W,:] .= J2  * ISz[1] + J1y * ISz[2] #4y
        M[8+6W,:,7+6W,:] .= J1y * ISz[1] + J2  * ISz[2] #3y

        # W nearest neighbor
        M[8+6W,:,1+ W,:] .= J2 *  ISx[1] + J1x * ISx[3] #4x
        M[8+6W,:,2+2W,:] .= J1x * ISx[1] + J2  * ISx[3] #2x
        M[8+6W,:,3+3W,:] .= J2 *  ISy[1] + J1x * ISy[3] #4x
        M[8+6W,:,4+4W,:] .= J1x * ISy[1] + J2  * ISy[3] #2x
        M[8+6W,:,5+5W,:] .= J2 *  ISz[1] + J1x * ISz[3] #4x
        M[8+6W,:,6+6W,:] .= J1x * ISz[1] + J2  * ISz[3] #2x

        # W+1 nearest neighbor
        M[8+6W,:,2+ W,:] .= J2 * ISx[1]
        M[8+6W,:,4+3W,:] .= J2 * ISy[1]
        M[8+6W,:,6+5W,:] .= J2 * ISz[1]

        # W-1 nearest neighbor
        M[8+6W,:,1+2W,:] .= J2 * ISx[3]
        M[8+6W,:,3+4W,:] .= J2 * ISy[3]
        M[8+6W,:,5+6W,:] .= J2 * ISz[3]
    end
    return M
end

MPO_2x2(model::J1J2) = MPO_2x2(J1xJ1yJ2(model.S, model.W, model.J1, model.J1, model.J2))