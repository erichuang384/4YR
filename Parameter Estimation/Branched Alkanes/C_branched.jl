using CMAEvolutionStrategy, Statistics, Random, DataFrames, Plots, LaTeXStrings, CSV, StaticArrays, Clapeyron
function IB_viscosity_C(model::EoSModel, P, T, z = StaticArrays.SA[1.0];
        ξ_i = Dict("CH3" => 0.507593380610382, "CH2" => 0.0462438047643044, "CH" => -0.42564193918, "C" => 0.5))

    """
    Overall Viscosity using method proposed by Ian Bell, 3 parameters
    """
    n_g = [2.544060456,	-1.926681797,	0.592109738] # global parameters
    gamma = 1.3016771263302

    # ensure z is an AbstractVector for broadcasting below
    zvec = collect(z)

    ξ_pure = zeros(length(zvec))

    # compute ξ for each pure component (make lookup robust)
    for j in 1:length(zvec)
        ξ_val = 0.0
        groups = model.groups.groups[j]       # expected Vector{String}
        num_groups = model.groups.n_groups[j] # expected Vector{Int}

        # Defensive: ensure groups and num_groups have same length
        if length(groups) != length(num_groups)
            error("groups and n_groups length mismatch for component $j")
        end

        for i in 1:length(groups)
            g = groups[i]
            # Use get to safely lookup Dict entries
            xi = get(ξ_i, g, nothing)
            if xi === nothing
                # missing group key -> return NaN or throw. Better: signal to caller by returning NaN
                @warn "Missing ξ_i entry for group" group=g component_index=j
                return fill(NaN, length(T))  # propagate invalid predictions
            end
            ξ_val += xi * num_groups[i]
        end

        ξ_pure[j] = ξ_val
    end

    # mixture ξ (dot product)
    ξ_mix = sum(zvec .* ξ_pure)

    R = Rgas()
    s_res = entropy_res(model, P, T, z)
    #s_red = -s_res ./ R
    s_red = ((-s_res ./ R) ^ gamma) ./ log(T)

    # guard against zero ξ_mix or non-finite s_red
    if !isfinite(ξ_mix) || ξ_mix == 0.0 || any(!isfinite, s_red)
        return fill(NaN, length(T))
    end

    ln_n_reduced = n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) .+
                   n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) .+
                   n_g[3] .* (s_red ./ ξ_mix) .^ (2.8)

    n_reduced = exp.(ln_n_reduced) .- 1.0

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar .* N_A

    Mw = Clapeyron.molecular_weight(model, z)
    m = Mw / N_A

    # elementwise sqrt and powers
    n_res = (n_reduced .* (ρ_N .^ (2/3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2/3))

    if length(zvec) == 1
        viscosity = IB_CE(model, T) .+ n_res
    else
        viscosity = IB_CE_mix(model, T, z) .+ n_res
    end

    return viscosity
end

# ==== Optimization Algorithm
Base.exit_on_sigint(false)

# === Objective Function  ===function make_global_objective(models::Vector, datasets::Vector{DataFrame})
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [xi_CH]  (in your simplified example)
    """
    function objective(x)
        # construct ξ_i using the optimization variables
        ξ_i = Dict("CH3" => 0.507593380610382, "CH2" => 0.0462438047643044, "CH" => x[1], "C" => x[2])

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = IB_viscosity_C.(model, Pvals[:], Tvals[:]; ξ_i = ξ_i)

                # If IB_viscosity_CH returned NaNs because of missing group keys etc.
                if any(x -> !isfinite(x), μ_pred) || any(x -> !isfinite(x), μ_exp)
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
    lower = [-5.0, -5.0],
    upper = [ 5.0, 5.0],
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
    SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    #SAFTgammaMie(["2-methylnonane"]), NOT TP
    #SAFTgammaMie(["4-methylnonane"]), NOT TP
    SAFTgammaMie(["heptamethylnonane"]),
    #SAFTgammaMie(["squalane"])
]

data_paths = [
    "Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
    #"Training DATA/Branched Alkane/2-methylnonane.csv",
    #"Training DATA/Branched Alkane/4-methylnonane.csv",
    "Training DATA/Branched Alkane/heptamethylnonane.csv",
    #"Training DATA/Branched Alkane/squalane.csv"
]


datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization
res = estimate_xi_CH3_CH2_CMA!(
    models,
    datasets;
    lower =  [-10.0, -10.0],
    upper =  [5.0,  5.0],
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))