# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    SkewArnoldi{T,TH,TV}

Result of a skew-orthogonal Arnoldi factorization.

Contains the upper Hessenberg matrix `H` and basis vectors `V` such that
`A * V[:, 1:k] = V * H` where `V` is a (skew-)orthonormal basis with respect
to the given skew-orthogonal inner product.

# Fields
- `H`: Upper Hessenberg matrix ((k+1)×k) - includes subdiagonal with residual norm
- `V`: Basis vectors matrix (n×(k+1)) - columns are the orthonormal basis vectors

The last row of `H` contains only `H[k+1, k]`, the residual norm after k steps.
This is useful for computing residuals, error estimates, and continuing the iteration.
"""
struct SkewArnoldi{T, TH<:AbstractMatrix{T}, TV<:AbstractMatrix{T}}
    H::TH  # Upper Hessenberg matrix ((k+1)×k)
    V::TV  # Basis vectors (n×(k+1))
end

"""
    skew_arnoldi!(op!, inner, v, k; [H], [V], [tmp])

Compute `k` steps of the Arnoldi iteration using a modified symplectic
Gram-Schmidt process with respect to a skew-orthogonal inner product.

# Arguments
- `op!(w, v)`: In-place operator that computes `w = A * v`
- `inner(u, v)`: Skew-orthogonal inner product returning `⟨u, v⟩_J`
  (e.g., `u' * J * v` for symplectic J)
- `v`: Starting vector (will be normalized in-place)
- `k`: Number of Arnoldi iterations

# Keyword Arguments
- `H`: Pre-allocated upper Hessenberg matrix of size `(k+1) × k`. If not provided,
  will be allocated.
- `V`: Pre-allocated matrix of size `(n, k+1)` to store the orthonormal basis vectors.
  If not provided, will be allocated.
- `tmp`: Pre-allocated temporary vector of length `n`. If not provided, will be allocated.

# Returns
A `SkewArnoldi` struct containing the upper Hessenberg matrix `H[1:k, 1:k]`.

# Notes
The algorithm uses symplectic Gram-Schmidt orthogonalization, processing basis vectors
in Darboux pairs `(eᵢ, fᵢ)` where odd columns are `eᵢ` and even columns are `fᵢ`.
For each pair, the projection formula is: `v_⊥ = v - Ω(eᵢ, v)·fᵢ + Ω(fᵢ, v)·eᵢ`.

Normalization uses the symplectic inner product for `fᵢ` vectors (ensuring `Ω(eᵢ, fᵢ) = 1`)
and the standard 2-norm for `eᵢ` vectors (since `Ω(v, v) = 0` always).

The inner product function should satisfy `⟨u, v⟩_J = -⟨v, u⟩_J` for skew-symmetry.
"""
function skew_arnoldi!(
    op!::F1,
    inner::F2,
    v::AbstractVector{T},
    k::Integer;
    H::AbstractMatrix{T} = zeros(T, k+1, k),
    V::AbstractMatrix{T} = similar(v, length(v), k+1),
    tmp::AbstractVector{T} = similar(v)
) where {T, F1, F2}

    n = length(v)
    k ≥ 1 || throw(ArgumentError("k must be ≥ 1"))
    k ≤ n || throw(ArgumentError("k must be ≤ n = $n"))
    size(H) == (k+1, k) || throw(DimensionMismatch("H must be $(k+1) × $k"))
    size(V) == (n, k+1) || throw(DimensionMismatch("V must be $n × $(k+1)"))
    length(tmp) == n || throw(DimensionMismatch("tmp must have length $n"))

    # Normalize starting vector with respect to skew inner product
    # For skew inner product ⟨v, v⟩_J = 0 always, so we use regular norm
    β = norm(v)
    iszero(β) && throw(ArgumentError("Starting vector must be nonzero"))
    @. V[:, 1] = v / β

    # Fill H with zeros
    fill!(H, zero(T))

    @inbounds for j = 1:k
        # w = A * v_j (stored in tmp)
        @views op!(tmp, V[:, j])

        # Modified symplectic Gram-Schmidt orthogonalization
        # Orthogonalize against complete Darboux pairs (e_i, f_i) = (V[:, 2i-1], V[:, 2i])
        # Using formula: v_⊥ = v - Ω(e,v)·f + Ω(f,v)·e
        n_pairs = div(j, 2)
        @views for i = 1:n_pairs
            e_idx = 2i - 1
            f_idx = 2i
            h_e = inner(V[:, e_idx], tmp)  # Ω(e_i, tmp)
            h_f = inner(V[:, f_idx], tmp)  # Ω(f_i, tmp)
            # tmp = tmp - h_e * f_i + h_f * e_i
            axpy!(-h_e, V[:, f_idx], tmp)
            axpy!(h_f, V[:, e_idx], tmp)
            # Store in H: coefficients for reconstruction A*V ≈ V*H
            H[e_idx, j] = -h_f
            H[f_idx, j] = h_e
        end

        # Compute norm of the residual for the subdiagonal
        # Use symplectic normalization: for f vectors (even j+1), normalize so Ω(e, f) = 1
        # For e vectors (odd j+1), use regular 2-norm since Ω(v, v) = 0
        if iseven(j + 1)
            # j+1 is even, so tmp will be f_{(j+1)/2}, paired with e = V[:, j]
            @views H[j+1, j] = inner(V[:, j], tmp)
        else
            # j+1 is odd, so tmp will be a new e vector - use 2-norm
            H[j+1, j] = norm(tmp)
        end

        # Check for breakdown (lucky breakdown or invariant subspace found)
        if abs(H[j+1, j]) < eps(real(T)) * max(one(real(T)), norm(H[1:j, j]))
            # Early termination - return what we have (j steps completed)
            return SkewArnoldi(H[1:j+1, 1:j], V[:, 1:j+1])
        end

        # Normalize and store the new basis vector
        @views V[:, j+1] .= tmp ./ H[j+1, j]
    end

    return SkewArnoldi(H, V)
end

"""
    skew_arnoldi!(op!, J::JMatrix, v, k; kwargs...)

Convenience method that uses the standard J-inner product `⟨u, v⟩_J = u' * J * v`
where `J` is a `JMatrix` (block-diagonal with [0 1; -1 0] blocks).
"""
function skew_arnoldi!(
    op!::F,
    J::JMatrix{TJ, SGN},
    v::AbstractVector{T},
    k::Integer;
    kwargs...
) where {F, T, TJ, SGN}

    # J-inner product: ⟨u, v⟩_J = u' * J * v
    function j_inner(u::AbstractVector, w::AbstractVector)
        # Compute u' * J * w efficiently using J's structure
        # J has blocks [0 1; -1 0] on the diagonal
        acc = zero(promote_type(eltype(u), eltype(w)))
        n = length(u)
        @inbounds for i = 1:2:n-1
            # J * w at positions i, i+1 is [w[i+1], -w[i]] * SGN
            acc += u[i] * (SGN * w[i+1])
            acc += u[i+1] * ((-SGN) * w[i])
        end
        return acc
    end

    return skew_arnoldi!(op!, j_inner, v, k; kwargs...)
end

"""
    skew_arnoldi(A, J, v, k)

Non-mutating version that creates a standard operator from matrix `A`.

# Arguments
- `A`: Matrix or linear operator
- `J`: `JMatrix` defining the skew inner product, or a function `inner(u, v)`
- `v`: Starting vector
- `k`: Number of Arnoldi iterations

# Returns
A `SkewArnoldi` struct containing the upper Hessenberg matrix `H`.
"""
function skew_arnoldi(A::AbstractMatrix{T}, J, v::AbstractVector{T}, k::Integer) where T
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    length(v) == n || throw(DimensionMismatch("v must have length $n"))

    # Create in-place operator
    tmp_op = similar(v)
    function mat_op!(w::AbstractVector, x::AbstractVector)
        mul!(w, A, x)
        return nothing
    end

    # Pre-allocate workspace
    H = zeros(T, k+1, k)
    V = similar(v, n, k+1)
    tmp = similar(v)
    v_copy = copy(v)

    return skew_arnoldi!(mat_op!, J, v_copy, k; H=H, V=V, tmp=tmp)
end

"""
    symplectic_mgs!(V, inner, j, coeffs)

Perform one step of modified symplectic Gram-Schmidt orthogonalization.
Orthogonalizes `V[:, j+1]` against complete Darboux pairs in `V[:, 1:j]`.

Uses the symplectic projection formula: for each pair (e_i, f_i) where Ω(e_i, f_i) = 1,
    v_⊥ = v - Ω(e_i, v)·f_i + Ω(f_i, v)·e_i

Odd columns are e_i vectors, even columns are f_i vectors.
coeffs[2i-1] stores -Ω(f_i, v), coeffs[2i] stores Ω(e_i, v).

This is an internal helper function.
"""
function symplectic_mgs!(
    V::AbstractMatrix{T},
    inner::F,
    j::Integer,
    coeffs::AbstractVector{T}
) where {T, F}

    n_pairs = div(j, 2)
    @inbounds @views for i = 1:n_pairs
        e_idx = 2i - 1
        f_idx = 2i
        h_e = inner(V[:, e_idx], V[:, j+1])  # Ω(e_i, v)
        h_f = inner(V[:, f_idx], V[:, j+1])  # Ω(f_i, v)
        # v = v - h_e * f_i + h_f * e_i
        axpy!(-h_e, V[:, f_idx], V[:, j+1])
        axpy!(h_f, V[:, e_idx], V[:, j+1])
        coeffs[e_idx] = -h_f
        coeffs[f_idx] = h_e
    end

    return nothing
end

"""
    skew_arnoldi_reorthog!(op!, inner, v, k; [H], [V], [tmp], [reorthog_tol])

Arnoldi iteration with modified symplectic Gram-Schmidt and selective reorthogonalization.

Similar to `skew_arnoldi!`, but performs a second orthogonalization pass when
loss of orthogonality is detected. This improves numerical stability at the
cost of additional computation.

# Additional Keyword Arguments
- `reorthog_tol`: Tolerance for triggering reorthogonalization. Default is `sqrt(eps(T))`.
  Reorthogonalization is performed when `‖h_new - h_old‖ / ‖h_old‖ > reorthog_tol`.
"""
function skew_arnoldi_reorthog!(
    op!::F1,
    inner::F2,
    v::AbstractVector{T},
    k::Integer;
    H::AbstractMatrix{T} = zeros(T, k+1, k),
    V::AbstractMatrix{T} = similar(v, length(v), k+1),
    tmp::AbstractVector{T} = similar(v),
    coeffs::AbstractVector{T} = similar(v, k),
    reorthog_tol::Real = sqrt(eps(real(T)))
) where {T, F1, F2}

    n = length(v)
    k ≥ 1 || throw(ArgumentError("k must be ≥ 1"))
    k ≤ n || throw(ArgumentError("k must be ≤ n = $n"))
    size(H) == (k+1, k) || throw(DimensionMismatch("H must be $(k+1) × $k"))
    size(V) == (n, k+1) || throw(DimensionMismatch("V must be $n × $(k+1)"))
    length(tmp) == n || throw(DimensionMismatch("tmp must have length $n"))
    length(coeffs) ≥ k || throw(DimensionMismatch("coeffs must have length ≥ $k"))

    # Normalize starting vector
    β = norm(v)
    iszero(β) && throw(ArgumentError("Starting vector must be nonzero"))
    @. V[:, 1] = v / β

    fill!(H, zero(T))

    @inbounds for j = 1:k
        # w = A * v_j
        @views op!(tmp, V[:, j])

        # First MGS pass - symplectic paired orthogonalization
        n_pairs = div(j, 2)
        @views for i = 1:n_pairs
            e_idx = 2i - 1
            f_idx = 2i
            h_e = inner(V[:, e_idx], tmp)
            h_f = inner(V[:, f_idx], tmp)
            axpy!(-h_e, V[:, f_idx], tmp)
            axpy!(h_f, V[:, e_idx], tmp)
            H[e_idx, j] = -h_f
            H[f_idx, j] = h_e
        end

        # Compute norm before potential reorthogonalization
        h_norm_old = norm(@view H[1:j, j])

        # Second MGS pass (reorthogonalization) if needed - compute corrections
        @views for i = 1:n_pairs
            e_idx = 2i - 1
            f_idx = 2i
            h_e = inner(V[:, e_idx], tmp)
            h_f = inner(V[:, f_idx], tmp)
            coeffs[e_idx] = -h_f
            coeffs[f_idx] = h_e
        end

        # Check if reorthogonalization is needed
        corr_norm = norm(@view coeffs[1:j])
        if corr_norm > reorthog_tol * max(h_norm_old, one(real(T)))
            @views for i = 1:n_pairs
                e_idx = 2i - 1
                f_idx = 2i
                H[e_idx, j] += coeffs[e_idx]
                H[f_idx, j] += coeffs[f_idx]
                # Apply correction: tmp = tmp - (-h_f)*e - h_e*f = tmp + h_f*e - h_e*f
                # But coeffs stores -h_f and h_e, so:
                axpy!(-coeffs[e_idx], V[:, e_idx], tmp)  # subtract (-h_f)*e = add h_f*e
                axpy!(-coeffs[f_idx], V[:, f_idx], tmp)  # subtract h_e*f
            end
        end

        # Subdiagonal entry - use symplectic normalization for f vectors (even j+1)
        if iseven(j + 1)
            # j+1 is even, so tmp will be f_{(j+1)/2}, paired with e = V[:, j]
            @views H[j+1, j] = inner(V[:, j], tmp)
        else
            # j+1 is odd, so tmp will be a new e vector - use 2-norm
            H[j+1, j] = norm(tmp)
        end

        # Check for breakdown
        if abs(H[j+1, j]) < eps(real(T)) * max(one(real(T)), norm(H[1:j, j]))
            # Early termination - return what we have (j steps completed)
            return SkewArnoldi(H[1:j+1, 1:j], V[:, 1:j+1])
        end

        # Normalize
        @views V[:, j+1] .= tmp ./ H[j+1, j]
    end

    return SkewArnoldi(H, V)
end

"""
    skew_lanczos!(op!, inner, v, k; [T_subdiag], [V], [tmp])

Compute `k` steps of the Lanczos iteration for skew-symmetric operators
with a skew-orthogonal inner product.

For skew-symmetric operators with a compatible inner product, the Hessenberg
matrix reduces to a tridiagonal form. This specialized routine exploits this
structure for improved efficiency.

# Arguments
- `op!(w, v)`: In-place skew-symmetric operator
- `inner(u, v)`: Skew-orthogonal inner product returning a scalar
- `v`: Starting vector
- `k`: Number of Lanczos iterations

# Keyword Arguments
- `T_subdiag`: Pre-allocated sub/super-diagonal (length `k`)
- `V`: Pre-allocated basis matrix `(n, k+1)`
- `tmp`: Pre-allocated temporary vector

# Returns
A `SkewHermTridiagonal` matrix representing the projected operator.
"""
function skew_lanczos!(
    op!::F1,
    inner::F2,
    v::AbstractVector{T},
    k::Integer;
    T_subdiag::AbstractVector{T} = zeros(T, k),
    V::AbstractMatrix{T} = similar(v, length(v), k+1),
    tmp::AbstractVector{T} = similar(v)
) where {T<:Real, F1, F2}

    n = length(v)
    k ≥ 1 || throw(ArgumentError("k must be ≥ 1"))
    k ≤ n || throw(ArgumentError("k must be ≤ n = $n"))
    size(V) == (n, k+1) || throw(DimensionMismatch("V must be $n × $(k+1)"))
    length(tmp) == n || throw(DimensionMismatch("tmp must have length $n"))
    length(T_subdiag) ≥ k || throw(DimensionMismatch("T_subdiag must have length ≥ $k"))

    # Normalize starting vector
    β = norm(v)
    iszero(β) && throw(ArgumentError("Starting vector must be nonzero"))
    @. V[:, 1] = v / β

    fill!(T_subdiag, zero(T))
    β_prev = zero(T)

    @inbounds for j = 1:k
        # w = A * v_j
        @views op!(tmp, V[:, j])

        # For skew-symmetric operator, α_j = ⟨v_j, A v_j⟩_J = 0
        # (since operator is skew and inner product is skew)

        # Orthogonalize against previous two vectors only (3-term recurrence)
        if j > 1
            # tmp = tmp - β_{j-1} * v_{j-1}
            @views axpy!(-β_prev, V[:, j-1], tmp)
        end

        # β_j = ‖tmp‖
        β_j = norm(tmp)

        # Check for breakdown
        if abs(β_j) < eps(real(T)) * max(one(real(T)), abs(β_prev))
            return SkewHermTridiagonal(T_subdiag[1:j-1])
        end

        T_subdiag[j] = β_j

        if j < k
            # Normalize
            @views V[:, j+1] .= tmp ./ β_j
        end

        β_prev = β_j
    end

    return SkewHermTridiagonal(T_subdiag[1:k])
end
