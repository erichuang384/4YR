function mix_xi(model::EoSModel, z::Vector{Float64}, ξ::Dict{String, Float64}, α::Dict{Tuple{String, String}, Float64}=Dict())
    names = model.components
    N = length(z)

    # Linear (ideal) contribution
    ξ0 = sum(z[i] * ξ[names[i]] for i in 1:N)

    # Cross (excess) contribution
    Δξ = 0.0
    for i in 1:(N-1), j in (i+1):N
        name_i, name_j = names[i], names[j]
        ξi, ξj = ξ[name_i], ξ[name_j]
        αij = get(α, (name_i, name_j), get(α, (name_j, name_i), 0.0))
        Δξ += αij * z[i] * z[j] * (ξi - ξj)
    end

    return ξ0 + Δξ
end
