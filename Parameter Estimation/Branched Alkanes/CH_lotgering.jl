using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays
#include("model development.jl")
function make_bell_objective(models::Vector, datasets::Vector{DataFrame})
    function objective(x)
        # unpack optimization vector
        n_alpha = Dict(
            "CH3" => (-0.0152941746, -0.26365184, -0.933877108),
            "CH2" => (-0.0000822187, -0.275395668, -0.2219383),
            "CH"  => (x[1], x[2], x[3])
        )
        params = Dict(
            "n_alpha" => n_alpha,
            "gamma"   => 0.41722646,
            "D_i"       => -0.1693155
        )

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = bell_lot_viscosity_opt.(model, Pvals[:], Tvals[:]; params=params)
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
    lower = [-1, -100, -1],
    upper = [ 1,  100, 150],
    seed = 42, σ0 = 0.5, max_iters = 8000
)
    Random.seed!(seed)
    obj = make_bell_objective(models, datasets)

    # initial guess
    #x0 = [-0.0068188, -0.02899088, -1.6803e-9, -0.000505772, -0.019999, -9.996391e-10, 0.05499806, -0.014602705977361975]

    x0 = [-0.014654678281817133, -0.23473496214139827, -0.9020364131672174]
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
    #SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    #SAFTgammaMie(["2-methylnonane"]), NOT TP
    #SAFTgammaMie(["4-methylnonane"]), NOT TP
    #SAFTgammaMie(["heptamethylnonane"]),
    #SAFTgammaMie(["squalane"])
]

data_paths = [
    #"Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
    #"Training DATA/Branched Alkane/2-methylnonane.csv",
    #"Training DATA/Branched Alkane/4-methylnonane.csv",
    #"Training DATA/Branched Alkane/heptamethylnonane.csv",
    #"Training DATA/Branched Alkane/squalane.csv"
]


datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization

# [-0.014654678281817133, -0.23473496214139827, -0.9020364131672174, -0.0003560000328943363, -0.25005253284627926, -0.23199855085571386, 0.3987245320443016, -0.16933433660430597]
res = optimize_bell_parameters!(
    models,
    datasets;
   lower  = [-1.0, -1.0, -1.0],

    upper  = [2.0, 0.5, 2.0],
    seed = 42,
    σ0 = 1.0,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))