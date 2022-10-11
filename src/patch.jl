using CUDA

_arraytype(::CuArray) = CuArray
_arraytype(  ::Array) =   Array