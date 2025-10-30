using CMAEvolutionStrategy
using Statistics, Random
using DataFrames
using Plots
using LaTeXStrings, CSV, StaticArrays, Clapeyron

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
        #ξ_T = Dict("CH3" => x[3], "CH2" => x[4])
        #A = x[1]
        #B = x[1]
        n_g_1 = x[3]
        n_g_2 = x[4]
        n_g_3 = x[5]
        n_g_4 = x[6]
        #n_exp = x[6]

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = IB_viscosity_TP.(model, Pvals[:], Tvals[:]; ξ_i = ξ_i, n_g_1 = n_g_1, n_g_2 = n_g_2, n_g_3 = n_g_3, n_g_4 = n_g_4)

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
    lower = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    upper = [0.6, 1.0, 1.0, 1.0, 1.0, 1.0],
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
res = estimate_xi_CH3_CH2_CMA!(
    models,
    datasets;
    lower =  [0.5, 0.0, -5.0, -5.0, -5.0, -5.0],
    upper =  [3.5, 1.0,  5.0, 5.0,  5.0,  5.0],
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))
println("Best objective value: ", fbest(res))
#=
best_x = xbest(res)

ξ_i = Dict("CH3" => best_x[1], "CH2" => best_x[2])
A, B = best_x[3], best_x[4]
n_g = best_x[5:7]
μ_pred = IB_viscosity_TP(models[1], 1e5, 300; ξ_i=ξ_i, A=A, B=B, n_g_1=n_g[1], n_g_2=n_g[2], n_g_3=n_g[3])


models = [
    SAFTgammaMie(["Butane"]),
    SAFTgammaMie(["Pentane"]),
    SAFTgammaMie(["Hexane"]),
    SAFTgammaMie(["Heptane"]),
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie([("Nonane",["CH3"=>2,"CH2"=>7])]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie([("Undecane",["CH3"=>2,"CH2"=>9])]),
    SAFTgammaMie([("Dodecane",["CH3"=>2,"CH2"=>10])]),
    SAFTgammaMie([("Tridecane",["CH3"=>2,"CH2"=>11])]),
    SAFTgammaMie([("Tetradecane",["CH3"=>2,"CH2"=>12])]),
    SAFTgammaMie([("Pentadecane",["CH3"=>2,"CH2"=>13])]),
    SAFTgammaMie([("Hexadecane",["CH3"=>2,"CH2"=>14])]),
    SAFTgammaMie([("Heptadecane",["CH3"=>2,"CH2"=>15])])
]

#experimental values
exp_nonane = CSV.read("Validation Data/Nonane DETHERM.csv", DataFrame)
exp_undecane = CSV.read("Validation Data/Undecane DETHERM.csv", DataFrame)
exp_tridecane = CSV.read("Validation Data/Tridecane DETHERM.csv", DataFrame)
exp_pentadecane = CSV.read("Validation Data/Pentadecane DETHERM.csv", DataFrame)
exp_heptadecane = CSV.read("Validation Data/Heptadecane DETHERM.csv", DataFrame)

exp_butane = CSV.read("Training Data/Butane DETHERM.csv", DataFrame)
exp_pentane =  CSV.read("Training Data/Pentane DETHERM.csv", DataFrame)
exp_hexane =  CSV.read("Training Data/Hexane DETHERM.csv", DataFrame)
exp_heptane =  CSV.read("Training Data/Heptane DETHERM.csv", DataFrame)
exp_octane = CSV.read("Training Data/Octane DETHERM.csv", DataFrame)
exp_decane =  CSV.read("Training Data/Decane DETHERM.csv", DataFrame)
exp_dodecane =  CSV.read("Training Data/Dodecane DETHERM.csv", DataFrame)
exp_tetradecane =  CSV.read("Training Data/Tetradecane DETHERM.csv", DataFrame)
exp_hexadecane =  CSV.read("Training Data/Hexadecane DETHERM.csv", DataFrame)


exp_data = [exp_butane, exp_pentane, exp_hexane, exp_heptane ,exp_octane, exp_nonane, exp_decane, exp_undecane, exp_dodecane, exp_tridecane, exp_tetradecane, exp_pentadecane, exp_hexadecane, exp_heptadecane] 

AAD = zeros(length(models))

#AAD_pentadecane = zeros(length(exp_pentadecane[:,1]))
for i in 1:length(models)
    T_exp = exp_data[i][:,2]
    n_exp = exp_data[i][:,3]
    P_exp = exp_data[i][:,1] 
    n_calc = IB_viscosity_TP.(models[i],P_exp,T_exp) 

    AAD[i] = sum(abs.( (n_exp .- n_calc)./n_exp))/length(P_exp)
end
println("AAD = ", AAD)
=#