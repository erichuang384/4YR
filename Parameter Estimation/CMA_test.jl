using CMAEvolutionStrategy
using Statistics, Random
using DataFrames
using Plots
using LaTeXStrings, CSV, StaticArrays

#include("bell_functions.jl")
#include("temp_bell_optimization.jl")
#include("Parameter Estimation/optimization_functions.jl")
# Allow Ctrl+C to interrupt instead of killing Julia
Base.exit_on_sigint(false)


# === Objective Function (same as before) ===
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [xi_CH3, xi_CH2, xiT_CH3, xiT_CH2, n_g_3]
    """
    function objective(x)
        ξ_i = Dict("CH3" => x[1], "CH2" => x[2])
        ξ_T = Dict("CH3" => x[3], "CH2" => x[4])
        n_g_3 = x[5]
        #n_exp = x[6]

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = IB_viscosity_global_2_6.(model, Pvals[:], Tvals[:];
                    ξ_i = ξ_i, ξ_T = ξ_T, n_g_3 = n_g_3)

                if any(!isfinite, μ_pred)
                    total_error += 1e10
                    continue
                end

                total_error += sum(((μ_exp .- μ_pred) ./ μ_exp).^2) / length(Pvals)

            catch err
                @warn "Invalid point encountered during optimization" x = x error = err
                total_error += 1e10
            end
        end

        return isfinite(total_error) ? total_error : 1e10
    end
    return objective
end


# === CMA-ES Optimization ===
function estimate_xi_CH3_CH2_CMA!(models::Vector, datasets::Vector{DataFrame};
    lower = [0.4, 0.02, -1.0, -1.0, 0.02],
    upper = [0.6, 0.08, 0.0, 0.1, 0.08],
    seed = 42, σ0 = 0.1, max_iters = 5000)

    Random.seed!(seed)
    obj = make_global_objective(models, datasets)

    # Initial guess: midpoint of bounds
    x0 = (lower .+ upper) ./ 2

    println("Starting CMA-ES optimization (xi_CH3, xi_CH2, ...) — seed=$seed")
    println("Initial guess: ", x0)

    iter_counter = Ref(0)

    result = minimize(
        obj,
        x0,
        σ0;
        lower = lower,
        upper = upper,
        seed = seed,
        verbosity = 2,
        maxiter = max_iters,
        ftol = 1e-9,
        callback = (opt, x, fx, ranks) -> begin
            iter_counter[] += 1
            if iter_counter[] % 20 == 0
                println("Iter $(iter_counter[]): fmin=$(minimum(fx)) best=$(xbest(opt))")
            end
        end
    )


    println("\nCMA-ES optimization complete.")
    println("Best parameters found:")
    println(xbest(result))
    println("Objective value = ", fbest(result))

    return result
end


# === Example Usage ===
models = [
    SAFTgammaMie(["Butane"]),
    SAFTgammaMie(["Hexane"]),
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie(["Dodecane"]),
    SAFTgammaMie(["Tetradecane"]),
    SAFTgammaMie(["Hexadecane"])
]

data_paths = [
    "Training DATA/Butane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Training DATA/Tetradecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv"
]

datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization
res = estimate_xi_CH3_CH2_CMA!(
    models,
    datasets;
    lower =  [0.207*0.95, 0.0049*0.95, -0.18/0.95, -0.89/0.95, 0.0278*0.95],
    upper =  [0.207/0.95, 0.0049/0.95, -0.18*0.95, -0.89*0.95, 0.0278/0.95],
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))
println("Best objective value: ", fbest(res))
