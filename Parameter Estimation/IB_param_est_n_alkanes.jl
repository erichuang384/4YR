using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings
using Metaheuristics
using Statistics, Random, StaticArrays

include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))
include("optimization_functions.jl")

# === Objective Function for Multiple Models ===cd
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [xi_CH3, xi_CH2]
    """
    function objective(x)
        xi_GCM = Dict("CH3" => x[1], "CH2" => x[2])
        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            μ_pred = IB_viscosity_GCM_xi_3param.(model, Pvals[:], Tvals[:]; xi_GCM = xi_GCM)

            # sum of squared relative errors
            total_error += sum(abs.(μ_exp .- μ_pred)./μ_exp)./length(Pvals)
        end

        return total_error
    end
    return objective
end


# === Global Optimization ===

function estimate_xi_CH3_CH2!(models::Vector, datasets::Vector{DataFrame};
                              lower = [0.0, 0.0], upper = [2.0, 2.0],
                              seed = 1234, max_iters = 5000)

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

    println("Starting global optimization (xi_CH3, xi_CH2) — bounds: $(lower)–$(upper), seed=$(seed), max_iters=$(max_iters)")

    state = Metaheuristics.optimize(obj, bounds, method; logger = logger)

    result = Metaheuristics.get_result(method)
    println("\nOptimization complete.")
#    println("Best (xi_CH3, xi_CH2) = ", result.best_sol)
#    println("Objective = ", result.best_f)

    return result
end

# Define models and corresponding data (can be 1..N)
models = [
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie(["Dodecane"]),
    SAFTgammaMie(["Tetradecane"]),
    SAFTgammaMie(["Hexadecane"])
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
res = estimate_xi_CH3_CH2!(models, datasets; lower = [0.3, 0.02], upper = [0.5, 0.1], seed = 42, max_iters = 5000)
println(res)
println("Best (xi_CH3, xi_CH2) = ", res.best_sol)
