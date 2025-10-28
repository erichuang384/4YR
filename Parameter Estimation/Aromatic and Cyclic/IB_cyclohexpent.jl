using CMAEvolutionStrategy
using Statistics, Random
using DataFrames
using Plots
using LaTeXStrings, CSV, StaticArrays, Clapeyron

#
#include("temp_bell_optimization.jl")
#include("Parameter Estimation/optimization_functions.jl")
# Allow Ctrl+C to interrupt instead of killing Julia
Base.exit_on_sigint(false)

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
# === Objective Function (same as before) ===
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [xi_CH3, xi_CH2, xiT_CH3, xiT_CH2, n_g_3]
    """
    function objective(x)
        ξ =  x[1]
        C_i =  x[2]

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = IB_pure_const.(model, Pvals[:], Tvals[:]; ξ = ξ, C_i = C_i)
                if any(!isfinite, μ_pred)
                    total_error += 1e10
                    continue
                end

                total_error = sum(((μ_exp .- μ_pred) ./ μ_exp).^2) / length(Pvals)

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
    lower = [0.4, 0.0],
    upper = [0.6,1.0],
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
models = [SAFTgammaMie(["cyclohexane"])]


data_paths = ["Training DATA/Cyclohexane DETHERM.csv"]

datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization
res = estimate_xi_CH3_CH2_CMA!(
    models,
    datasets;
    lower =  [0.0,-10.0],
    upper =  [1.5, 10.0],
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))
println("Best objective value: ", fbest(res))
