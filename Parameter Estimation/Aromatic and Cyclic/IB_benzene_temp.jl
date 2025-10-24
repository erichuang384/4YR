using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, Metaheuristics
using Statistics
using Random
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))
include("optimization_functions.jl")

model = SAFTgammaMie([("Benzene",["aCH"=>6])])


# === Objective Function for Multiple Models ===
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    ξ
    """
    function objective(x)
        ξ = x[1]
        ξ_T = x[2]
        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            μ_pred = IB_3param_T_pure_optimize.(model, Pvals[:], Tvals[:]; ξ = ξ, ξ_T = ξ_T)

            # sum of squared relative errors
            total_error = sum(((μ_exp .- μ_pred)./μ_exp) .^ 2)
            #total_error += sum((μ_exp .- μ_pred).^2)
        end

        return total_error
    end
    return objective
end


# === Global Optimization ===

function estimate_ξ!(models::Vector, datasets::Vector{DataFrame};
                              lower = [0.0, -10], upper = [2.0, 10],
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

    println("Starting global optimization (ξ) — bounds: $(lower)–$(upper), seed=$(seed), max_iters=$(max_iters)")

    state = Metaheuristics.optimize(obj, bounds, method; logger = logger)

    result = Metaheuristics.get_result(method)
    println("\nOptimization complete.")

    return result
end
models = [model]
data_paths = [
    "Training DATA/Benzene DETHERM.csv"
]

datasets = [load_experimental_data(p) for p in data_paths]

# Run global estimation
res = estimate_ξ!(models, datasets; lower = [1, 50.0], upper = [1.5, 100.0], seed = 42, max_iters = 5000)
println(res)
println("Best ξ = ", res.best_sol)
