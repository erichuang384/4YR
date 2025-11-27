using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays, StatsBase
include("model development.jl")
function make_bell_objective(models::Vector, datasets::Vector{DataFrame}; limit::Int=0)
    function objective(x)
        # unpack optimization vector
        n_alpha = Dict(
            "CH3" => (x[1],	x[2],	x[3]),
            "CH2" => (x[4],	x[5],	x[6]),
        )

        tau_i = Dict(
            "CH3" => (x[7]),
            "CH2" => (x[8])
        )

        params = Dict(
            "n_alpha" => n_alpha,
            "tau_i" => tau_i,
            "gamma"   => 0.45,
            "D_i" => x[9]
        )

        total_error = 0.0

        # === Loop over each model/dataset pair ===
        for (model, data) in zip(models, datasets)
            npoints = nrow(data)

            # --- Subsample if needed ---
            if limit > 0 && npoints > limit
                subset_idx = sample(1:npoints, limit; replace=false)
                subset = data[subset_idx, :]
            else
                subset = data
            end

            Pvals = subset.p
            Tvals = subset.t
            μ_exp = subset.viscosity

            try
                μ_pred = bell_lot_viscosity_opt_ogref.(model, Pvals[:], Tvals[:]; params=params)

                if any(!isfinite, μ_pred)
                    total_error += 1e10
                    continue
                end

                total_error += sum(((μ_exp .- μ_pred) ./ μ_exp).^2) / length(Pvals)

            catch err
                @warn "Error during evaluation" error=err
                total_error += 1e10
            end
        end

        return isfinite(total_error) ? total_error : 1e10
    end
    return objective
end

function optimize_bell_parameters!(models, datasets;
    limit::Int=0,
    lower = fill(0.0, 9),
    upper = fill(1.0, 9),
    seed = 42,
     σ0 = 1.0, max_iters = 8000
)
    Random.seed!(seed)
    obj = make_bell_objective(models, datasets; limit=limit)

    # initial guess
    #x0 = [-0.0068188, -0.02899088, -1.6803e-9, -0.000505772, -0.019999, -9.996391e-10, 0.05499806, -0.014602705977361975]

    x0 =  [-0.005736064295840372, 3.3035558966216567, -9.949108171466614, -0.003482098937990341, 1.0230124247770518, -3.269146600001073, 1.4234261974321183, 0.035739755709452, 13.16452096627834]
# at 0.1686
    println("Starting CMA-ES optimization with seed = $seed")
    println("Initial parameters: ", x0)

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

    println("\nOptimization complete.")
    println("Best parameters found:")
    println(xbest(result))
    println("Objective value = ", fbest(result))

    return result
end

# === Example Usage ===
models = [
    SAFTgammaMie(["Pentane"]),
    SAFTgammaMie(["Hexane"]),
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie(["Dodecane"]),
    SAFTgammaMie(["Tridecane"]),
    SAFTgammaMie(["Pentadecane"]),
    SAFTgammaMie(["Hexadecane"])
]

data_paths = [
    "Training DATA/Pentane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Validation DATA/Tridecane DETHERM.csv",
    "Validation DATA/Pentadecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv"
]


datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization

# [-0.009906338133508082, -0.6770805839304224, -2.1715739702027115, 0.0010175814867223187, -0.9682046689892875, 0.32629202910983224, 0.5000000023041724, 0.5000000023292258, 0.5441397922288526, -0.2999999988321849]
lower_bounds = [-1.0, -1.0, -20.0, -0.1, -1.0, -5.0, 0.0, -0.2, 0.0]
upper_bounds = [0.0,  5.0,  0.0,  1.0,  5.0,  1.0, 1.5, 1.0, 20.0]
res = optimize_bell_parameters!(
    models,
    datasets;
    limit = 200, 
    lower  = lower_bounds,
    upper  = upper_bounds,
    seed = 42,
    σ0 = 3.0,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))