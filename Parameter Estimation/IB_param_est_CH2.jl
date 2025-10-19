using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, Metaheuristics
using Statistics
using Random
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))
include("optimization_functions.jl")

function IB_viscosity_GCM_with_ximap(model::EoSModel, P, T; xi_map = Dict("CH3" => 0.476193, "CH2" => 0.045))
    """
    Overall viscosity using method proposed by Ian Bell, but with xi_map passed in.
    """
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674]

    # Compute ξ
    ξ = 0.0
    groups = model.groups.groups[1]
    num_groups = model.groups.n_groups[1]
    for i in 1:length(groups)
        g = groups[i]
        if haskey(xi_map, g)
            ξ += xi_map[g] * num_groups[i]
        else
            error("xi_map missing entry for group \"$g\".")
        end
    end

    R = Clapeyron.R̄
    s_res = entropy_res(model, P, T)
    s_red = -s_res ./ R

    n_reduced = exp.(n_g[1] .* (s_red ./ ξ) .+ n_g[2] .* (s_red ./ ξ).^1.5 .+
                     n_g[3] .* (s_red ./ ξ).^2 .+ n_g[4] .* (s_red ./ ξ).^2.5) .- 1

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A
    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A

    n_res = (n_reduced .* (ρ_N .^ (2/3)) .* sqrt.(m .* k_B .* T)) ./ (s_red .^ (2/3))
    μ = IB_CE(model, T) .+ n_res
    return μ
end

function load_experimental_data(path::AbstractString)
    df = CSV.read(path, DataFrame)
    rename!(df, lowercase.(names(df)))  # standardize lowercase column names
    return df
end

function make_objective(models::Vector, datasets::Vector{DataFrame}; xi_ch3 = 0.476193)
    function objective(x)
        xi_ch2 = x[1]  # single variable optimization
        xi_map = Dict("CH3" => xi_ch3, "CH2" => xi_ch2)
        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            μ_pred = IB_viscosity_GCM_with_ximap.(Ref(model), Pvals, Tvals; xi_map = xi_map)
            total_error += mean(((μ_exp .- μ_pred) ./ μ_exp).^2)
        end

        return total_error
    end
    return objective
end

function estimate_xi_CH2!(models::Vector, datasets::Vector{DataFrame};
                          lower = 0.0, upper = 2.0,
                          seed = 1234, max_iters = 5000)
    rng = MersenneTwister(seed)
    Random.seed!(rng)

    obj = make_objective(models, datasets)
    bounds = hcat(lower, upper)'

    method = DE()
    method.options = Options(iterations = max_iters, f_calls_limit = 1_000_000,
                             store_convergence = true, seed = seed)

    logger = function (status)
        if isdefined(status, :iteration)
            if status.iteration % 50 == 0 || status.iteration == 1
                println("iter=$(status.iteration) f_calls=$(status.f_calls) best_sol=$(status.best_sol)")
            end
        end
    end

    println("Starting optimization (xi_CH2) — bounds: [$lower, $upper], seed=$seed, max_iters=$max_iters")
    state = Metaheuristics.optimize(obj, bounds, method; logger = logger)

    result = Metaheuristics.get_result(method)
    return result
end

# Define models and corresponding data
models = [
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie([("Dodecane", ["CH3"=>2, "CH2"=>10])]),
    SAFTgammaMie([("Tetradecane", ["CH3"=>2, "CH2"=>12])]),
    SAFTgammaMie([("Hexadecane", ["CH3"=>2, "CH2"=>14])])
]

data_paths = [
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Training DATA/Tetradecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv"
]

datasets = [load_experimental_data(p) for p in data_paths]

# Run global estimation
res = estimate_xi_CH2!(models, datasets; lower = 0.0, upper = 1.0, seed = 42, max_iters = 5000)
println(res)
println("Best (xi_CH2) = ", res.best_sol)
