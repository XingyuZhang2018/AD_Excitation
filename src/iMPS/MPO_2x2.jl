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

contract4(x) = ein"((ae,bf),cg),dh->abcdefgh"(x...)
I_S(S) = (d = size(S, 1); Id = I(d); 
         [reshape(contract4(circshift([S, Id, Id, Id], i)), (d^4,d^4)) for i in 0:3]
        )
I_4(d) = (Id = I(d); reshape(contract4([Id, Id, Id, Id]), (d^4,d^4)))
H_on_site(S) = (d = size(S, 1); Id = I(d); 
                reshape(mapreduce(contract4, +,
                                  [[ S,  S, Id, Id], 
                                   [ S, Id,  S, Id], 
                                   [Id,  S, Id,  S], 
                                   [Id, Id,  S,  S]]
                                 ), (d^4,d^4))
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
