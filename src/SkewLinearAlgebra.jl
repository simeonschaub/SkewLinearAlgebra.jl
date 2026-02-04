# This file is a part of Julia. License is MIT: https://julialang.org/license
"""
This module based on the LinearAlgebra module provides specialized functions
and types for skew-symmetricmatrices, i.e A=-A^T
"""
module SkewLinearAlgebra

using LinearAlgebra
import LinearAlgebra as LA
export
    #Types
    SkewHermitian,
    SkewHermTridiagonal,
    SkewCholesky,
    SkewCholeskyNoPivot,
    JMatrix,
    SkewArnoldi,
    #functions
    isskewhermitian,
    isskewsymmetric,
    skewhermitian,
    skewhermitian!,
    pfaffian,
    pfaffian!,
    logabspfaffian,
    logabspfaffian!,
    skewchol,
    skewchol!,
    skew_arnoldi,
    skew_arnoldi!,
    skew_arnoldi_reorthog!,
    skew_lanczos!

include("skewhermitian.jl")
include("tridiag.jl")
include("jmatrix.jl")
include("hessenberg.jl")
include("skeweigen.jl")
include("eigen.jl")
include("exp.jl")
include("cholesky.jl")
include("pfaffian.jl")
include("arnoldi.jl")
end



