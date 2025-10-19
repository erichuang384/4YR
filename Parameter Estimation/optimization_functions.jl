function IB_viscosity_GCM_xi(model::EoSModel, P, T; xi_GCM = Dict("CH3" => 0.484458, "CH2" => 0.047793))
    """
    Function for Optimization of Bell method for GCM xi values
    """
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674]

    # Compute ξ from groups
    ξ = 0.0
    groups = model.groups.groups[1]
    num_groups = model.groups.n_groups[1]
    for i in 1:length(groups)
        g = groups[i]
        if !haskey(xi_GCM, g)
            error("xi_GCM missing entry for group \"$g\".")
        end
        ξ += xi_GCM[g] * num_groups[i]
    end

    R = Clapeyron.R̄
    s_res = entropy_res(model, P, T)
    s_red = -s_res ./ R

    n_reduced = exp.(n_g[1] .* (s_red ./ ξ) .+ n_g[2] .* (s_red ./ ξ) .^ (1.5) .+
                     n_g[3] .* (s_red ./ ξ) .^ 2 .+ n_g[4] .* (s_red ./ ξ) .^ (2.5)) .- 1

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A

    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A

    n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))
    μ = IB_dilute_gas_viscosity(model, T) .+ n_res
    return μ
end

function IB_viscosity_pure_xi(model::EoSModel, P, T; ξ = 1)
    """
    Function for Optimization of Bell method for GCM xi values
    """
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674]

    R = Clapeyron.R̄
    s_res = entropy_res(model, P, T)
    s_red = -s_res ./ R

    n_reduced = exp.(n_g[1] .* (s_red ./ ξ) .+ n_g[2] .* (s_red ./ ξ) .^ (1.5) .+
                     n_g[3] .* (s_red ./ ξ) .^ 2 .+ n_g[4] .* (s_red ./ ξ) .^ (2.5)) .- 1

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A

    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A

    n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))
    μ = IB_CE(model, T) .+ n_res
    return μ
end

function IB_viscosity_pure_xi_3param(model::EoSModel, P, T; ξ = 1)
    """
    Function for Optimization of Bell method for GCM xi values
    """
    n_g = [0.301491667, -0.143158667, 0.036576]

    R = Clapeyron.R̄
    s_res = entropy_res(model, P, T)
    s_red = -s_res ./ R

    n_reduced = exp(n_g[1] .* (s_red ./ ξ) .^ (1.8) + n_g[2] .* (s_red ./ ξ) .^ (2.4) + n_g[3] .* (s_red ./ ξ) .^ (2.8)) - 1

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A

    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A

    n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))
    μ = IB_CE(model, T) .+ n_res
    return μ
end

function load_experimental_data(path::AbstractString)
    """
    Format experimental data from CSV
    """
    df = CSV.read(path, DataFrame)
    # normalize column names to lowercase symbols
    rename!(df, Symbol.(lowercase.(String.(names(df)))))
    required = [:p, :t, :viscosity]
    #for c in required
    #    @assert c ∈ names(df) "Missing column: $c in $path. Expected columns: $(required)."
    #end
    return df
end