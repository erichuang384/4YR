using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays, StatsBase, Printf
#include("model development.jl")  # must provide load_experimental_data, etc.

global R = Rgas()
global N_A = Clapeyron.N_A
global k_B = Clapeyron.k_B
# === Your model function (unchanged), expects params Dict with "n_alpha", "gamma", "D_i" ===
function bell_lot_viscosity_opt_s_idref(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # Extract optimization parameters
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    γ       = params["gamma"]::Float64
    D_i     = params["D_i"]::Float64  # single global d passed in

    # Model & group data
    groups      = model.groups.groups[1]
    num_groups  = model.groups.n_groups[1]
    S           = model.params.shapefactor.values
    σ           = diag(model.params.sigma.values).* 1e10  # m → Å

    # Volume contribution
    V = sum(num_groups.* S.* (σ.^ 3))

    # Compute group contributions
    n_g_matrix = zeros(length(groups), 3)
    for (i, grp) in enumerate(groups)
        @assert haskey(n_alpha, grp) "Missing n_alpha entry for group '$grp'"
        Aα, Bα, Cα = n_alpha[grp]
        n_g_matrix[i, 1] = Aα * S[i] * σ[i]^3 * num_groups[i]
        n_g_matrix[i, 2] = Bα * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = Cα * num_groups[i]
    end

    n_g = vec(sum(n_g_matrix, dims = 1))
    m_gc = sum(num_groups)  # total group count

    # Thermodynamic terms
    #R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    s_red = -s_res / R
    s_id  = entropy_ideal(model, P, T, z)

    # Entropy-based reduced coordinate (your chosen form)
    z_term = (-s_res / (s_id) + log(-s_res / R) / m_gc)

    Mw = Clapeyron.molecular_weight(model, z)

    # Reduced viscosity correlation (polynomial in z_term with global d)
    D = D_i * m_gc
    ln_n_reduced = n_g[1] + n_g[2]*z_term + n_g[3]*z_term^2 + D*z_term^3
    n_reduced = exp(ln_n_reduced) - 1.0

    # Physical constants
    #N_A = Clapeyron.N_A
    #k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    m = Mw / N_A

    # Residual contribution (Ian Bell scaling)
    n_res = n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T) / (s_red ^ (2/3))

    # Overall viscosity
    viscosity = if length(z) == 1
        IB_CE(model, T) + n_res
    else
        IB_CE_mix(model, T, z) + n_res
    end

    return viscosity
end

# === Objective: globals (CH3/CH2 A,B,C; gamma) + single global d (shared across datasets) ===
function make_bell_objective_globalD(models::Vector, datasets::Vector{DataFrame}; limit::Int=0)
    function objective(params::AbstractVector{<:Real})
        # Expect params = [CH3(A,B,C), CH2(A,B,C), gamma, d_global]
        if length(params) != 8
            return 1e20
        end

        # Unpack globals
        A_CH3, B_CH3, C_CH3 = params[1], params[2], params[3]
        A_CH2, B_CH2, C_CH2 = params[4], params[5], params[6]
        gamma               = params[7]
        d_global            = params[8]

        # Build the global n_alpha Dict
        n_alpha = Dict(
            "CH3" => (A_CH3, B_CH3, C_CH3),
            "CH2" => (A_CH2, B_CH2, C_CH2),
        )

        total_error = 0.0

        # Loop over each model/dataset
        for (model, data) in zip(models, datasets)
            npoints = nrow(data)
            subset = if limit > 0 && npoints > limit
                idx = sample(1:npoints, limit; replace=false)
                data[idx, :]
            else
                data
            end

            Pvals = subset.p
            Tvals = subset.t
            μ_exp = subset.viscosity

            # Per-call params dict: share globals + single d
            params_dict = Dict(
                "n_alpha" => n_alpha,
                "gamma"   => gamma,
                "D_i"     => d_global,
            )

            try
                # To further optimize, parallelize over data points if many
                μ_pred = Vector{Float64}(undef, length(Pvals))
                Threads.@threads for i in 1:length(Pvals)
                    μ_pred[i] = bell_lot_viscosity_opt_s_idref(model, Pvals[i], Tvals[i]; params=params_dict)
                end

                if any(!isfinite, μ_pred)
                    total_error += 1e10
                    continue
                end

                # normalized SSE per dataset (your style)
                total_error += sum(((μ_exp.- μ_pred)./ μ_exp).^2) / length(Pvals)

            catch err
                @warn "Error during evaluation" error=err
                total_error += 1e10
            end
        end

        return isfinite(total_error) ? total_error : 1e10
    end
    return objective
end

# === Optimizer wrapper (globals + single global d) ===
function optimize_bell_parameters_globalD!(models::Vector{<:EoSModel}, datasets::Vector{DataFrame};
    limit::Int=0,
    seed::Int=42,
    σ0::Float64=0.8,
    max_iters::Int=10000,
    # Bounds for globals (7): CH3(A,B,C), CH2(A,B,C), gamma
    lower_globals::Vector{Float64} = [-5.0, -5.0, -20.0, -5.0, -15.0, -15.0, -1.0],
    upper_globals::Vector{Float64} = [ 5.0,  5.0,  20.0,  5.0,  15.0,  15.0,  1.0],
    # Bounds for single global d
    lower_D::Float64 = -20.0,
    upper_D::Float64 =  20.0,
    # Initial guess (globals + single d)
    x0_globals::Vector{Float64} = [-0.0162489433762928, -1.30116529162194, -13.215313775983,
                                   0.00029301076673254, -1.01191765765676, -2.9913861277034,
                                   0.43779367539929],
    x0_D::Float64 = -7.17608578256157
)
    Random.seed!(seed)

    # Build objective
    obj = make_bell_objective_globalD(models, datasets; limit=limit)

    # Build initial vector and bounds
    x0    = vcat(x0_globals, x0_D)
    lower = vcat(lower_globals, lower_D)
    upper = vcat(upper_globals, upper_D)

    println("Starting CMA-ES optimization (globals + single d) — seed=$seed")
    println("Initial parameters: ", x0)

    iter_counter = Ref(0)

    result = minimize(
        obj,
        x0,
        σ0;
        lower = lower,
        upper = upper,
        seed  = seed,
        verbosity = 2,  # Reduce verbosity for faster output
        multi_threading = true,  # Enable multi-threaded parallel evaluation
        #popsize = 4 + floor(Int, 3*log(length(x0))),
        #stagnation = 100 + 100 * length(x0)^1.5/(4 + floor(Int, 3*log(length(x0)))),
        maxiter = max_iters,
        ftol = 1e-9,
        callback = (opt, x, fx, ranks) -> begin
            iter_counter[] += 1
            if iter_counter[] % 20 == 0
                try
                    println("Iter $(iter_counter[]): fmin=$(minimum(fx)) best=$(xbest(opt))")
                catch
                    println("Iter $(iter_counter[]): callback invoked (no xbest/fmin available)")
                end
            end
        end
    )

    println("\nOptimization complete.")
    best = xbest(result)
    fmin = fbest(result)

    # Unpack best parameters
    A_CH3, B_CH3, C_CH3 = best[1], best[2], best[3]
    A_CH2, B_CH2, C_CH2 = best[4], best[5], best[6]
    gamma               = best[7]
    d_global            = best[8]

    println("Best globals found:")
    @printf "  CH3: A = %.8f, B = %.8f, C = %.8f\n" A_CH3 B_CH3 C_CH3
    @printf "  CH2: A = %.8f, B = %.8f, C = %.8f\n" A_CH2 B_CH2 C_CH2
    @printf "  gamma = %.8f\n" gamma
    @printf "  d (global) = %.8f\n" d_global
    println("Objective value = ", fmin)

    return best, fmin, result
end

# === Example usage: add back your commented models/datasets ===
models = [
    SAFTgammaMie(["Pentane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Octane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Decane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Pentadecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = WalkerIdeal),
#	SAFTgammaMie(["Eicosane"], idealmodel = WalkerIdeal)
]


data_paths = [
    "Training Data/Pentane DETHERM.csv",
    "Training Data/Hexane DETHERM.csv",
    "Training Data/Octane DETHERM.csv",
    "Training Data/Decane DETHERM.csv",
    "Training Data/Dodecane DETHERM.csv",
    "Validation Data/Tridecane DETHERM.csv",
   "Validation Data/Pentadecane DETHERM.csv",
    "Training Data/Hexadecane DETHERM.csv",
#	"Training DATA/n-eicosane.csv"
]


datasets = [load_experimental_data(p) for p in data_paths]

best, fmin, status = optimize_bell_parameters_globalD!(
    models, datasets;
    limit = 500,
    seed = 42,
    σ0 = 1.5,
    max_iters = 10000,
    lower_globals = [-0.1, -3.0, -15.0, -0.01, -5.0, -4.0, 0.0],
    upper_globals = [0.1,  15.0,  5.0,  0.01,  10.0,  5.0, 1.5],
    lower_D = -2.0,
    upper_D =  15.0,
    x0_globals = [0.005952848564039981, 4.9987620750887585, -5.363555061920793, 0.0005406463914980795, 2.2909984027918773, 2.4371834602304503, 1.0534485344121052],
    x0_D = 3.533496436518558
)

println("\nBest solution (CMA-ES): ", best)
println("Best objective value: ", fmin)