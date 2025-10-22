using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings
using Metaheuristics, Distributions
using Statistics, Random, StaticArrays

# === Objective Function for Multiple Models ===
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [a2, b2, c2]  (length 3, single group CH2)
    """
    function objective(x)
        if length(x) != 3
            error("optimizer vector x must have length 3: [a2, b2, c2]")
        end

        # CH3 parameters (already known)
        a_CH3 = -0.00617185
        b_CH3 = -0.194443
        c_CH3 = -0.0597023

        # CH2 parameters (to be optimized)
        params = Dict(
            :a_α => [a_CH3, x[1]],
            :b_α => [b_CH3, x[2]],
            :c_α => [c_CH3, x[3]],
        )

        total_error = 0.0
        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            μ_pred = [Lotgering_viscosity_optimize(model, Pvals[i], Tvals[i]; params = params)
                      for i in 1:length(Pvals)]

            total_error += sum(((μ_exp .- μ_pred) ./ μ_exp) .^ 2)
        end

        return total_error
    end
    return objective
end


# === Global Optimization ===
function estimate_params!(models::Vector, datasets::Vector{DataFrame};
                          lower = [-1e-2, -1e-3, -1.0e-1],
                          upper = [1e-2,  1e-3,  1.0e-1],
                          seed = 1234, max_iters = 5000)

    rng = MersenneTwister(seed)
    Random.seed!(rng)

    obj = make_global_objective(models, datasets)
    bounds = hcat(lower, upper)'

    # --- Initial guess (Lotgering default CH2 coefficients) ---
    x0 = [-0.00149404, -0.240412, 0]  # Default CH2 values from Lotgering

    # --- Differential Evolution setup with seeded population ---
    pop_random = (0.9 .+ 0.2 .* rand(19, length(x0))) .* x0'
    pop = vcat(x0', pop_random)  # ensure first candidate = x0

    method = DE(init_population = pop)
    method.options = Options(iterations = max_iters,
                             f_calls_limit = 1_000_000,
                             store_convergence = true,
                             seed = seed)

    # --- Logger ---
    logger = function (status)
        if hasproperty(status, :iteration)
            if status.iteration % 50 == 0 || status.iteration == 1
                bestf = hasproperty(status, :best_f) ? status.best_f :
                        (hasproperty(status, :best) && hasproperty(status.best, :f) ? status.best.f : NaN)
                bestx = hasproperty(status, :best_sol) ? status.best_sol :
                        (hasproperty(status, :best) && hasproperty(status.best, :sol) ? status.best.sol : [])
                println("iter=$(status.iteration)  f_calls=$(status.f_calls)  best_f=$(bestf)")
                println("    current x = ", bestx)
            end
        end
    end

    println("Starting global optimization (CH2 params) — seed=$(seed), max_iters=$(max_iters)")
    println("Initial guess: ", x0)

    # --- Run optimization ---
    result = Metaheuristics.optimize(obj, bounds, method; logger = logger)

    # --- Extract best values ---
    best_x = hasproperty(result, :best_sol) ? result.best_sol :
              (hasproperty(result, :best) ? result.best.sol : missing)
    best_f = hasproperty(result, :best_f) ? result.best_f :
              (hasproperty(result, :best) ? result.best.f : missing)

    println("\nOptimization complete.")
    println("Best objective value = ", best_f)
    println("Best parameter vector (CH2) = ", best_x)

    return result
end


# === Prepare models and data ===

models = [
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie(["Dodecane"]),
    SAFTgammaMie(["Tetradecane"]),
    SAFTgammaMie(["Hexadecane"]),
]

data_paths = [
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Training DATA/Tetradecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv",
]

datasets = [load_experimental_data(p) for p in data_paths]

# === Run global estimation ===
lower_bounds = [-1, -1, -2]
upper_bounds = [1e-2, 0.2, 0.2]
#x = [-0.001555382, -0.244068, -0.00589621] f = 1.3866
res = estimate_params!(models, datasets;
                       lower = lower_bounds,
                       upper = upper_bounds,
                       seed = 42,
                       max_iters = 2000)

println(res)
