using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays
include("model development.jl")
function make_bell_objective(models::Vector, datasets::Vector{DataFrame})
    function objective(x)
        # unpack optimization vector
        n_alpha = Dict(
            #"CH3" => (-0.0152941746, -0.26365184, -0.933877108),
            #"CH2" => (-0.0000822187, -0.275395668, -0.2219383)
            "CH3" => (x[1], x[2], x[3]),
            "CH2" => (x[4], x[5], x[6])
            #"CH"  => (x[1], x[2], x[3])
        )
        params = Dict(
            "n_alpha" => n_alpha,
            #"gamma"   => 0.41722646,
            "gamma"   => x[7],
            "D_1"       => x[8],
            "D_2"       => x[9],
            "tau"       => x[10]
        )

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = bell_lot_viscosity_lnT.(model, Pvals[:], Tvals[:]; params=params)
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
    lower = [ -1,  -1,   -1,  -1,  -1,  -1,  -1,  -1, -1, -1],
    upper = [ 0.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    seed = 42, σ0 = 0.5, max_iters = 8000
)
    Random.seed!(seed)
    obj = make_bell_objective(models, datasets)

    # initial guess
    #x0 = [-0.0068188, -0.02899088, -1.6803e-9, -0.000505772, -0.019999, -9.996391e-10, 0.05499806, -0.014602705977361975]

    x0 =  [-0.01961179017943229, -0.30835077862007293, -1.1643179758298794, 0.0003437513891417228, -0.22585529683757002, -0.18307569889440561, 0.4031108164614755, -0.09999999950431465, -0.07889063319133222, 1.137990220517076]
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
#  [-0.0196117, -0.308350, -1.16 0.000343751, -0.225855, -0.18307, 0.40311, -0.1, -0.0788, 1.13]
res = optimize_bell_parameters!(
    models,
    datasets;
    lower  = [   -0.1,    -1.0,   -5.0, -1.0,   -3.0, -3.0, 0.0, -1.0, -0.5, 0.5],
    upper  = [    0.0,     0.0,    0.0,  0.1,    0.0,  0.0, 3.0,  0.1,  0.0, 1.5],
    seed = 42,
    σ0 = 1e-4,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))
