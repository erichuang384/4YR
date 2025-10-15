using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, Metaheuristics
using Statistics, Random

include("all_functions.jl")

# === 1. Core Function ===
function IB_viscosity_GCM_with_ximap(model::EoSModel, P, T; xi_map = Dict("CH3" => 0.5, "CH2" => 0.045))
    """
    Overall viscosity using Ian Bell method with custom xi_map.
    """
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674]

    # Compute ξ from groups
    ξ = 0.0
    groups = model.groups.groups[1]
    num_groups = model.groups.n_groups[1]
    for i in 1:length(groups)
        g = groups[i]
        if !haskey(xi_map, g)
            error("xi_map missing entry for group \"$g\".")
        end
        ξ += xi_map[g] * num_groups[i]
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


# === 2. Load Experimental Data ===
function load_experimental_data(path::AbstractString)
    df = CSV.read(path, DataFrame)
    rename!(df, Symbol.(lowercase.(String.(names(df)))))  # normalize column names
    required = [:p, :t, :viscosity]
    for c in required
        @assert c ∈ names(df) "Missing column: $c"
    end
    return df
end


# === 3. Objective Function for Multiple Models ===
function make_global_objective(models::Vector{EoSModel}, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [xi_CH3, xi_CH2]
    """
    function objective(x)
        xi_map = Dict("CH3" => x[1], "CH2" => x[2])
        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            μ_pred = IB_viscosity_GCM_with_ximap.(model, Pvals[:], Tvals[:]; xi_map = xi_map)

            # sum of squared relative errors
            total_error += sum(((μ_exp .- μ_pred) ./ μ_exp) .^ 2)
        end

        return total_error
    end
    return objective
end


# === 4. Global Optimization ===
function estimate_xi_CH3_CH2!(models::Vector{EoSModel}, datasets::Vector{DataFrame};
                              lower = [0.0, 0.0], upper = [2.0, 2.0],
                              seed = 1234, max_iters = 2000)

    rng = MersenneTwister(seed)
    Random.seed!(rng)

    obj = make_global_objective(models, datasets)

    bounds = hcat(lower, upper)'  # 2×2 matrix (rows = bounds)

    method = DE()
    method.options = Options(iterations = max_iters,
                             f_calls_limit = 1_000_000,
                             store_convergence = true,
                             seed = seed)

    logger = function (status)
        if isdefined(status, :iteration)
            if status.iteration % 50 == 0 || status.iteration == 1
                println("iter=$(status.iteration) f_calls=$(status.f_calls) best_sol=$(status.best_sol)")
            end
        end
    end

    println("Starting global optimization (xi_CH3, xi_CH2) — bounds: $lower–$upper, seed=$seed, max_iters=$max_iters")
    state = Metaheuristics.optimize(obj, bounds, method; logger = logger)

    result = Metaheuristics.get_result(method)
    println("\nOptimization complete.")
    println("Best (xi_CH3, xi_CH2) = ", result.best_sol)
    println("Objective = ", result.best_f)

    return result
end


# === 5. Example Usage ===
# Define models and corresponding data
models = [
    SAFTgammaMie(["Ethane"])
#    SAFTgammaMie(["Octane"]),
#    SAFTgammaMie(["Nonane"]),
#    SAFTgammaMie(["Decane"])
]

data_paths = [
    "Parameter Estimation/ethane.csv",
    "Parameter Estimation/octane.csv",
    "Parameter Estimation/nonane.csv",
    "Parameter Estimation/decane.csv"
]

datasets = [load_experimental_data(p) for p in data_paths]

# Run global estimation
res = estimate_xi_CH3_CH2!(models, datasets; lower = [0.0, 0.0], upper = [1.0, 1.0], seed = 42, max_iters = 1000)
println(res)
