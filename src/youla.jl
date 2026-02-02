# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    Youla{T,M<:AbstractMatrix{T},V<:AbstractVector{<:Real}}

Structure to store the Youla decomposition of a real skew-symmetric matrix.

For a real skew-symmetric matrix `A` (where `A == -transpose(A)`), the Youla decomposition
finds a unitary matrix `U` such that `transpose(U) * A * U` is in block-diagonal form
with 2×2 blocks `σᵢ * [0 1; -1 0]` on the diagonal.

# Fields
- `U::M`: Unitary matrix such that `transpose(U) * A * U` gives the Youla normal form
- `σ::V`: Vector of singular values (one per 2×2 block)

The relation `transpose(U) * A * U ≈ Youla_form` holds, where the Youla form is
block-diagonal with blocks `σᵢ * [0 1; -1 0]` for each `σᵢ` in `σ`.
"""
struct Youla{T,M<:AbstractMatrix{T},V<:AbstractVector{<:Real}}
    U::M  # Unitary matrix
    σ::V  # Singular values (one per 2x2 block)

    function Youla{T,M,V}(U, σ) where {T,M<:AbstractMatrix{T},V<:AbstractVector{<:Real}}
        LA.require_one_based_indexing(U)
        new{T,M,V}(U, σ)
    end
end

"""
    Youla(U, σ)

Construct a `Youla` structure from the unitary matrix `U` and the vector of
singular values `σ`.
"""
function Youla(U::AbstractMatrix{T}, σ::AbstractVector{<:Real}) where {T}
    return Youla{T,typeof(U),typeof(σ)}(U, σ)
end

Base.size(Y::Youla) = size(Y.U)
Base.size(Y::Youla, d::Integer) = size(Y.U, d)

"""
    Matrix(Y::Youla)

Reconstruct the original skew-symmetric matrix from its Youla decomposition.
"""
function Base.Matrix(Y::Youla{T}) where {T}
    n = size(Y.U, 1)
    D = youla_form(Y.σ, n, T)
    return Y.U * D * transpose(Y.U)
end

"""
    youla_form(σ, n, T=eltype(σ))

Construct the Youla normal form matrix: block-diagonal with 2×2 blocks `σᵢ * [0 1; -1 0]`.
"""
function youla_form(σ::AbstractVector{<:Real}, n::Integer, ::Type{T}=eltype(σ)) where {T}
    D = zeros(T, n, n)
    k = length(σ)
    for i in 1:k
        j = 2i - 1
        D[j, j+1] = σ[i]
        D[j+1, j] = -σ[i]
    end
    return D
end

"""
    youla(A) -> Youla

Compute the Youla decomposition of a real skew-symmetric matrix `A`.

For a real skew-symmetric matrix `A` (where `A == -transpose(A)`), finds a unitary
matrix `U` such that `transpose(U) * A * U` is in Youla normal form: block-diagonal
with 2×2 blocks of the form `σᵢ * [0 1; -1 0]`, where `σᵢ` are the positive
singular values of `A` (each appearing once, not twice as in the standard SVD).

# Returns
A `Youla` structure with fields:
- `U`: Unitary matrix (complex for real input)
- `σ`: Vector of positive singular values (one per 2×2 block)

The decomposition satisfies: `transpose(Y.U) * A * Y.U ≈ youla_form(Y.σ, n)`

# Algorithm
The algorithm is based on the singular value decomposition. For real skew-symmetric
matrices, the singular values come in pairs, and the left and right singular
vectors can be combined to form the unitary matrix for the Youla form.

# References
- D.C. Youla, "A Normal Form for a Matrix under the Unitary Congruence Group",
  Canadian Journal of Mathematics, vol. 13, pp. 694-704, 1961.

# Examples
```julia
julia> A = [0.0 2.0 -7.0 4.0; -2.0 0.0 -8.0 3.0; 7.0 8.0 0.0 1.0; -4.0 -3.0 -1.0 0.0]
julia> Y = youla(A)
julia> transpose(Y.U) * A * Y.U  # Should be approximately in Youla form
```
"""
function youla(A::AbstractMatrix{<:Real})
    isskewsymmetric(A) || throw(ArgumentError("youla requires a skew-symmetric matrix (A == -transpose(A))"))
    return youla!(copyeigtype(A))
end

youla(A::SkewHermitian{<:Real}) = youla!(copyeigtype(A.data))

"""
    youla!(A) -> Youla

Same as [`youla`](@ref), but may overwrite `A` with intermediate computations.
"""
function youla!(A::AbstractMatrix{T}) where {T<:Real}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square"))

    # Handle trivial cases
    if n == 0
        return Youla(zeros(Complex{T}, 0, 0), T[])
    elseif n == 1
        return Youla(ones(Complex{T}, 1, 1), T[])
    end

    # Compute SVD: A = V * Σ * W'
    # For skew-symmetric A = -A^T, we have A = V*Σ*W' and A^T = W*Σ*V' = -A
    # This means W*Σ*V' = -V*Σ*W', implying W = -V*P for some permutation/sign P
    # when singular values are distinct.
    F = svd(A)
    S = F.S
    tol = eps(T) * max(n, 1) * (isempty(S) ? one(T) : maximum(S))

    CT = Complex{T}
    U = zeros(CT, n, n)
    σ = Vector{T}()

    k = 1  # column index in U
    i = 1  # index in singular values

    while i <= length(S) && k + 1 <= n
        if S[i] < tol
            break
        end

        # Get the singular vectors
        v = F.U[:, i]      # left singular vector
        w = F.V[:, i]      # right singular vector (column of V)

        # For a real skew-symmetric matrix with A*w = σ*v and A'*v = σ*w,
        # and using A' = -A, we get A*v = -σ*w
        #
        # We want U such that U^T * A * U = D where D has blocks σ*[0 1; -1 0]
        # This means for columns u1, u2 of U: u1^T * A * u2 = σ, u2^T * A * u1 = -σ
        #
        # Using u1 = (v - w)/√2, u2 = (v + w)/√2 gives the correct structure.

        u1_raw = v - w
        u2_raw = v + w

        norm1 = norm(u1_raw)
        norm2 = norm(u2_raw)

        if norm1 > tol && norm2 > tol
            u1 = complex.(u1_raw / norm1)
            u2 = complex.(u2_raw / norm2)

            # Gram-Schmidt to ensure exact orthogonality
            u2 = u2 - dot(u1, u2) * u1
            norm_u2 = norm(u2)
            if norm_u2 > tol
                u2 = u2 / norm_u2
            end

            U[:, k] = u1
            U[:, k+1] = u2
            push!(σ, S[i])
            k += 2
        end

        # Skip to next pair (singular values come in pairs for skew-symmetric)
        i += 2
    end

    # Fill null space columns
    if k <= n
        # Use remaining singular vectors for null space
        null_start = 2 * length(σ) + 1
        for j in null_start:n
            if k > n
                break
            end
            if j <= size(F.U, 2)
                U[:, k] = complex.(F.U[:, j])
            end
            k += 1
        end
    end

    # Ensure U is fully orthonormal
    if k <= n || !isapprox(U' * U, I, rtol=sqrt(tol))
        Q, _ = qr(U)
        U .= Matrix(Q)
    end

    return Youla(U, σ)
end

# Pretty printing
function Base.show(io::IO, mime::MIME"text/plain", Y::Youla)
    summary(io, Y)
    println(io)
    println(io, "Singular values:")
    show(io, mime, Y.σ)
    println(io)
    println(io, "Unitary factor U:")
    show(io, mime, Y.U)
end
