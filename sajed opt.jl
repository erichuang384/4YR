using CMAEvolutionStrategy, Statistics, Random, DataFrames, CSV, Clapeyron, StaticArrays, StatsBase

function tau_OFE(model::EoSModel, tau_i)
    """
    epsilon pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    x_sk_order = model.groups.flattenedgroups
    x_sk_dict = Dict(x_sk_order[i] => xₛₖ[i] for i in eachindex(x_sk_order))
    x_sk_vec   = [x_sk_dict[g] for g in x_sk_order] # ensure in same order
    tau_vec = [tau_i[g] for g in x_sk_order] # has to be matrix

    n = length(x_sk_order)
    tau_mat = zeros(n,n)
    for i in 1:n, j in 1:n
        tau_mat[i, j] = (i == j) ? tau_vec[i] : (tau_vec[i] + tau_vec[j])/2 # using mean value
    end

    #x_vec = model.groups.n_groups[1]./sum(model.groups.n_groups[1])
    #vdW mixing rule
    tau_ofe = (x_sk_vec' * (tau_mat.^1) * x_sk_vec)
    #tau_ofe = sum(tau_vec .* x_vec)
    return tau_ofe
end

function bell_lot_viscosity_opt_mgc(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # === Extract optimization parameters ===
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    γ       = params["gamma"]::Float64
    tau_i = params["tau_i"]
    D_i = params["D_i"]

    # === Model & group data ===
    groups      = model.groups.groups[1]
    num_groups  = model.groups.n_groups[1]
    S           = model.params.shapefactor
    σ           = diag(model.params.sigma.values) .* 1e10  # m → Å

    # === Volume contribution ===
    V = sum(num_groups .* S .* (σ .^ 3))

    # === Compute group contributions ===
    n_g_matrix = zeros(length(groups), 3)
#    tau = zeros(length(groups))

    for (i, grp) in enumerate(groups)
        @assert haskey(n_alpha, grp) "Missing n_alpha entry for group '$grp'"
        Aα, Bα, Cα = n_alpha[grp]
#        tau_dum = tau_i[grp]

        n_g_matrix[i, 1] = Aα * S[i] * σ[i]^3 * num_groups[i]
        n_g_matrix[i, 2] = Bα * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = Cα * num_groups[i]
#        tau[i] = tau_dum * num_groups[i]
    end

    n_g = vec(sum(n_g_matrix, dims = 1))
    #m_gc = sum(num_groups)
    m_gc = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

    tau_mix = tau_OFE(model,tau_i) # vdw

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    #z_term = (s_res) / (R * m_gc * log(T)^ tau_mix)
    z_term = m_gc * (-s_res) / (R * log(T)^ tau_mix)
    #s_red = -s_res / R

    Mw = Clapeyron.molecular_weight(model, z)

    # === Reduced viscosity correlation ===
    #D = (D_1+D_2/Mw)^(-1) #* m_gc
    D = D_i * m_gc
    ln_n_reduced = n_g[1] + n_g[2]*z_term + n_g[3]*z_term^2 + D*z_term^3
    n_reduced = exp(ln_n_reduced)

    # === Physical constants ===
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    
    m = Mw / N_A

    # === Residual contribution ===
    n_res = n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T)# / (z_term^(2/3))

    # === Chapman–Enskog or mixture viscosity ===
    viscosity = if length(z) == 1
        IB_CE(model, T) + n_res
    else
        IB_CE_mix(model, T, z) + n_res
    end

    return viscosity
end

function make_bell_objective(models::Vector, datasets::Vector{DataFrame})
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

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = bell_lot_viscosity_opt_mgc.(model, Pvals[:], Tvals[:]; params=params)
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
    lower = fill(0.0, 9),
    upper = fill(1.0, 9),
    seed = 42, σ0 = 0.5, max_iters = 8000
)
    Random.seed!(seed)
    obj = make_bell_objective(models, datasets)

    # initial guess
    #x0 = [-0.0068188, -0.02899088, -1.6803e-9, -0.000505772, -0.019999, -9.996391e-10, 0.05499806, -0.014602705977361975]

    x0 =   [-0.016248943,	1.301165292,	-13.21531378, 2.93E-04,	1.011917658,	-2.991386128, 0.94, 0.6, 7.14]

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
res = optimize_bell_parameters!(
    models,
    datasets;
    lower  = [-1.0, 0.0, -20.0, -0.1, -5.0, -10.0, 0.2, 0.2, 0.0],
    upper  = [0.0, 5.0,    0.0, 0.1, 5.0,     5.0, 1.3, 1.3, 15.0],
    seed = 42,
    σ0 = 0.5,
    max_iters = 10000
)

println("\nBest solution (CMA-ES): ", xbest(res))