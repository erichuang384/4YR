using CMAEvolutionStrategy, Statistics, Random, DataFrames, Plots, LaTeXStrings, CSV, StaticArrays, Clapeyron

function IB_viscosity_CH(model::EoSModel, P, T, z = StaticArrays.SA[1.0];
	ξ_i = Dict("CH3" => 2.09866287580761, "CH2" => 0.182534356794281, "CH" => -0.4))

	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [4.931211447,	-4.380364487,	1.569276144] # global parameters
	#ξ_pure = zeros(length(z))

    ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)
        ξ = 0

		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)

			xi = ξ_i[ξ_i[:, 1].==groups[i], 2][1]

			ξ = ξ + xi  * num_groups[i]

			ξ_pure[j] = ξ
		end

	end

	ξ_mix = sum(z .* ξ)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

    ln_n_reduced = n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8)
	n_reduced = exp(ln_n_reduced) .- 1.0 

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model, z)
	m = Mw / N_A

	n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))

	if length(z) == 1
		viscosity = IB_CE(model, T) + n_res
	else
		viscosity = IB_CE_mix(model, T, z) + n_res
	end

	return viscosity
end
# ==== Optimization Algorithm
Base.exit_on_sigint(false)


# === Objective Function  ===
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(x) where:
    x = [xi_CH3, xi_CH2, xiT_CH3, xiT_CH2, n_g_3]
    """
    function objective(x)
        ξ_i = Dict("CH3" => 2.09866287580761, "CH2" => 0.182534356794281, "CH" => x[1])

        total_error = 0.0

        for (model, data) in zip(models, datasets)
            Pvals = data.p
            Tvals = data.t
            μ_exp = data.viscosity

            try
                μ_pred = IB_viscosity_TP.(model, Pvals[:], Tvals[:]; ξ_i = ξ_i)

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
    lower = [-5.0],
    upper = [5.0],
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