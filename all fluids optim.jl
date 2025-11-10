using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays, StatsBase
function bell_lot_viscosity_all(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # === Extract optimization parameters ===
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    tau_i = params["tau_i"]
    D_i = params["D_i"]

    γ       = 0.45
    #=
    n_alpha = Dict(
            "CH3" => (-0.00839281621531174, -0.2536455462605472, -1.2079684228897345),
		    "CH2" => (0.0008160680272863482, -0.5540075192744599, 0.4562071725372469),
            "CH" => (0.03930050901260097, -1.5984961240693298, 10.093752915195292),
            "C"  => (0.19911712,	0.220305695,	1.292925164)
               )
    # molar frac
    tau_i = Dict(
        "CH3" => (0.24074570077425134),
		"CH2" => (0.5526998339960865),
        "CH" => (0.5604734970596769))
=#
    # === Model & group data ===
    groups      = model.groups.groups[1]
    num_groups  = model.groups.n_groups[1]
    S           = model.params.shapefactor
    σ           = diag(model.params.sigma.values) .* 1e10  # m → Å

    # === Volume contribution ===
    V = sum(num_groups .* S .* (σ .^ 3))

    # === Compute group contributions ===
    n_g_matrix = zeros(length(groups), 3)
    #tau = zeros(length(groups))

    for (i, grp) in enumerate(groups)
        @assert haskey(n_alpha, grp) "Missing n_alpha entry for group '$grp'"
        Aα, Bα, Cα = n_alpha[grp]
        #tau_dum = tau_i[grp]

        n_g_matrix[i, 1] = Aα * S[i] * σ[i]^3 * num_groups[i]
        n_g_matrix[i, 2] = Bα * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = Cα * num_groups[i]
        #tau[i] = tau_dum * num_groups[i]
    end

    n_g = vec(sum(n_g_matrix, dims = 1))
    m_gc = sum(num_groups)

    #tau_mix = sum(tau) ./ m_gc # makes it mole fraction based
    tau_mix = tau_OFE(model,tau_i)

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    z_term = (s_res) / (R * m_gc * log(T)^ tau_mix)
    s_red = -s_res / R

    Mw = Clapeyron.molecular_weight(model, z)

    # === Reduced viscosity correlation ===
    D = D_i * m_gc

    ln_n_reduced = n_g[1] + n_g[2]*z_term + n_g[3]*z_term^2 + D*z_term^3
    n_reduced = exp(ln_n_reduced) - 1.0

    # === Physical constants ===
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    
    m = Mw / N_A

    # === Residual contribution ===
    n_res = n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T) / (s_red^(2/3))

    # === Chapman–Enskog or mixture viscosity ===
    viscosity = if length(z) == 1
        IB_CE(model, T) + n_res
    else
        IB_CE_mix(model, T, z) + n_res
    end

    return viscosity
end


#include("model development.jl")
function make_bell_objective(models::Vector, datasets::Vector{DataFrame}; limit::Int=0)
    @assert length(models) == length(datasets) "models and datasets must have same length"

    function objective(x)
        # === Unpack optimization vector ===
        n_alpha = Dict(
            "CH3" => (x[1], x[2], x[3]),
            "CH2" => (x[4], x[5], x[6]),
            "CH"  => (x[7], x[8], x[9])
        )

        tau_i = Dict(
            "CH3" => x[10],
            "CH2" => x[11],
            "CH"  => x[12]
        )

        params = Dict(
            "n_alpha" => n_alpha,
            "tau_i"   => tau_i,
            "D_i"     => x[13]
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
                μ_pred = bell_lot_viscosity_all.(model, Pvals[:], Tvals[:]; params=params)

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
    lower = fill(0.0, 13),
    upper = fill(1.0, 13),
    seed = 42, σ0 = 0.5, max_iters = 8000
)
    Random.seed!(seed)
    obj = make_bell_objective(models, datasets; limit=limit)

    # initial guess
    #x0 = [-0.0068188, -0.02899088, -1.6803e-9, -0.000505772, -0.019999, -9.996391e-10, 0.05499806, -0.014602705977361975]

    x0 = [ -0.0162489433762928, -1.3, -13.3,
     0.00029301076673254,	-1.0,	-3.0,
     0.042101628,	-1.407726021,	11.03083133,
     0.08, 0.06, 0.08,
     -7.9]
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
    SAFTgammaMie(["Hexane"]),
    #SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    #SAFTgammaMie(["Dodecane"]),
    #SAFTgammaMie(["Tridecane"]),
    #SAFTgammaMie(["Pentadecane"]),
    SAFTgammaMie(["Hexadecane"]),
    #SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    #SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    #SAFTgammaMie(["2-methylnonane"]), NOT TP
    #SAFTgammaMie(["4-methylnonane"]), NOT TP
    #SAFTgammaMie(["heptamethylnonane"]),
    SAFTgammaMie(["squalane"])
]

data_paths = [
    #"Training DATA/Pentane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    #"Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    #"Training DATA/Dodecane DETHERM.csv",
    #"Validation DATA/Tridecane DETHERM.csv",
    #"Validation DATA/Pentadecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv",
    #"Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    #"Training DATA/Branched Alkane/2-methylbutane.csv",
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
    limit = 50, 
    #lower  = fill(-15.0, 13),
    lower = [-1.0, -5.0, -20.0, -0.1, -5.0, -10.0, 0.0, -5.0, 0.0, 0.0 , 0.0, 0.0, -15.0],
    upper = [0.0,   0.0,   0.0, 0.1,  0.1,   1.0,   1.0,  1.0, 15.0, 2.0, 2.0, 2.0, 2.0],
    #upper  = fill(15.0,13),
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))