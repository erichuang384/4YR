using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays
#include("model development.jl")
function make_bell_objective(models::Vector, datasets::Vector{DataFrame})
    function objective(x)
        # unpack optimization vector
        n_alpha = Dict(
            "CH3" => (-0.015087401852988442, 1.1793288696973334, -10.799687755027858),
            "CH2" => (-0.0006393202735293168, 0.9103543410756307, -1.9017722602334455),
            "CH"  => (x[1], x[2], x[3])
        )

        tau_i = Dict(
            "CH3" => (1.1714896072382803),
            "CH2" => (1.2),
            "CH"  => (x[4])
        )

        params = Dict(
            "n_alpha" => n_alpha,
            "tau_i" => tau_i,
            "gamma"   => 0.45,
            "D_i"       => 12.536185542576174
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
    lower = fill(0.0,4),
    upper = fill(0.0,4),
    seed = 42, σ0 = 0.5, max_iters = 8000
)
    Random.seed!(seed)
    obj = make_bell_objective(models, datasets)

    # initial guess
    #x0 = [-0.0068188, -0.02899088, -1.6803e-9, -0.000505772, -0.019999, -9.996391e-10, 0.05499806, -0.014602705977361975]

    x0 = [0.04, 1.4, 11.03, 0.7]
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
    #SAFTgammaMie(["Pentane"]),
    #SAFTgammaMie(["Hexane"]),
    #SAFTgammaMie(["Octane"]),
    #SAFTgammaMie(["Decane"]),
    #SAFTgammaMie(["Dodecane"]),
    #SAFTgammaMie(["Tridecane"]),
    #SAFTgammaMie(["Pentadecane"]),
    #SAFTgammaMie(["Hexadecane"]),
    #SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    #SAFTgammaMie(["2-methylnonane"]), NOT TP
    #SAFTgammaMie(["4-methylnonane"]), NOT TP
    #SAFTgammaMie(["heptamethylnonane"]),
    SAFTgammaMie(["squalane"])
]

data_paths = [
    #"Training DATA/Pentane DETHERM.csv",
    #"Training DATA/Hexane DETHERM.csv",
    #"Training DATA/Octane DETHERM.csv",
    #"Training DATA/Decane DETHERM.csv",
    #"Training DATA/Dodecane DETHERM.csv",
    #"Validation DATA/Tridecane DETHERM.csv",
    #"Validation DATA/Pentadecane DETHERM.csv",
    #"Training DATA/Hexadecane DETHERM.csv",
    #"Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
    #"Training DATA/Branched Alkane/2-methylnonane.csv",
    #"Training DATA/Branched Alkane/4-methylnonane.csv",
    #"Training DATA/Branched Alkane/heptamethylnonane.csv",
    "Training DATA/Branched Alkane/squalane.csv"
]


datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization

# [-0.014654678281817133, -0.23473496214139827, -0.9020364131672174, -0.0003560000328943363, -0.25005253284627926, -0.23199855085571386, 0.3987245320443016, -0.16933433660430597]
res = optimize_bell_parameters!(
    models,
    datasets;
   lower  = [-3.0, -5.0, -5.0, -2.0],

    upper  = [5.0, 150.5, 350.0, 2.0],
    seed = 42,
    σ0 = 0.5,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))