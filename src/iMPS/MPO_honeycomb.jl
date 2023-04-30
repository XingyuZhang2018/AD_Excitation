# """
#     MPO_honeycomb(model<:HamiltonianModel)

# site label
# ```
#   │       
# ──1       
#    \        │   
#     2───────1 
#     │        \    
#   ──1         2 ──
#      \        │       
#        2──────1 
#        │       \  
#                 2 ──
#                 │
# ```
# return the honeycomb helix MPO of the `model` as a 2-bond tensor.
# """

contract2(x) = (d = size(x[1],1); reshape(ein"ac,bd->abcd"(x...), (d^2,d^2)))

function MPO(model::Kitaev)
    S, W = model.S, model.W
    @assert W >= 2 "The width of the model must be at least 2."
    Sx, Sy, Sz = const_Sx(S), const_Sy(S), const_Sz(S)
    d = size(Sx, 1)

    M = zeros(ComplexF64, 3+W, d^2, 3+W, d^2)

    I2 = contract2([I(2), I(2)])
    M[  1,:,  1,:] .= I2
    M[3+W,:,3+W,:] .= I2

    M[  2,:,1,:] .= contract2([I(2), Sx  ])
    M[2+W,:,1,:] .= contract2([Sz  , I(2)])
    M[3+W,:,1,:] .= contract2([Sy  , Sy  ])
    for i in 2:W
        M[i+1,:,i,:] .= I2
    end
    M[3+W,:,1+W,:] .= contract2([Sx  , I(2)])
    M[3+W,:,2+W,:] .= contract2([I(2), Sz  ])
    return M
end