using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, Metaheuristics
using Statistics
using Random
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))
include("optimization_functions.jl")


function IB_viscosity_GCM_with_ximap(model::EoSModel, P, T; xi_map = Dict("CH3" => 0.5, "CH2" => 0.045, "aCH"=>0.5))
	"""
	Overall viscosity using method proposed by Ian Bell, but with xi_map passed in.
	"""
	# global parameters (unchanged)
	n_g = [-0.448046, 1.012681, -0.381869, 0.054674]

	# Compute ξ from groups in the model using xi_map
	ξ = 0.0
	groups = model.groups.groups[1] # n-element Vector{String} (per your code)
	num_groups = model.groups.n_groups[1] # n-element Vector{Int}
	for i in 1:length(groups)
		g = groups[i]
		if haskey(xi_map, g)
			value = xi_map[g]
		else
			error("xi_map missing entry for group \"$g\". Add it to xi_map or extend defaults.")
		end
		ξ += value * num_groups[i]
	end

	R = Clapeyron.R̄
	# ensure broadcasting works if inputs are vectors
	s_res = entropy_res(model, P, T)
	s_red = -s_res ./ R

	# compute reduced n
	n_reduced = exp.(n_g[1] .* (s_red ./ ξ) .+ n_g[2] .* (s_red ./ ξ) .^ (1.5) .+
					 n_g[3] .* (s_red ./ ξ) .^ 2 .+ n_g[4] .* (s_red ./ ξ) .^ (2.5)) .- 1

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model)
	m = Mw / N_A

	n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))

	μ = IB_CE(model, T) .+ n_res
	return μ
end

model = SAFTgammaMie(["Ethane"])
@show model

function load_experimental_data(path::AbstractString)
	df = CSV.read(path, DataFrame)
	required = [:P, :T, :viscosity]
	#missing_cols = setdiff(required, names(df))
	#if !isempty(missing_cols)
	#    error("Experimental CSV missing columns: $(join(missing_cols, \", \")). Expected $(required).")
	#end
	return df
end

function make_objective(model::EoSModel, data::DataFrame; xi_ch2 = 0.045)
	function objective(x)
		# x is a vector; we only optimize first element xi_CH3
		xi_ch3 = x[1]
		# create xi_map
		xi_map = Dict("CH3" => xi_ch3, "CH2" => xi_ch2)


		# compute predicted viscosities for all rows
		Pvals = data.p
		Tvals = data.T
		μ_pred = IB_viscosity_GCM_with_ximap.(model, Pvals[:], Tvals[:]; xi_map = xi_map)

		μ_exp = data.viscosity
		# ensure same shapes
		if length(μ_pred) != length(μ_exp)
			# attempt to broadcast or convert
			μ_pred = vec(μ_pred)
		end


		# compute RMSE
		objective_function = sum(((μ_exp .- μ_pred)./μ_exp) .^ 2)
		return objective_function
	end
	return objective
end

# --- Optimization wrapper ---
function estimate_xi_CH3!(model::EoSModel, data::DataFrame; lower = 0.0, upper = 2.0, seed = 1234, max_iters = 2000)
	rng = MersenneTwister(seed)
	Random.seed!(rng)


	obj = make_objective(model, data)


	# bounds matrix: first row lower bounds, second row upper bounds (2 x n) as required by Metaheuristics.jl
	bounds = [lower upper]'


	# pick algorithm and configure options
	method = DE() # Differential Evolution
	method.options = Options(iterations = max_iters, f_calls_limit = 1_000_000, store_convergence = true, seed = seed)


	# optional logger to print progress every 50 iterations
	logger = function (status)
		if isdefined(status, :iteration)
			if status.iteration % 50 == 0 || status.iteration == 1
				println("iter=$(status.iteration) f_calls=$(status.f_calls) best_sol=$(status.best_sol) ")
			end
		end
	end


	println("Starting optimization (xi_CH3) — bounds: [$lower, $upper], seed=$seed, max_iters=$max_iters")
	state = Metaheuristics.optimize(obj, bounds, method; logger = logger)


	# extract result
	result = Metaheuristics.get_result(method)

	#println("Optimization finished.")
	#println("Best xi_CH3 = ", result.best_sol)
	#println("Objective (RMSE) = ", result.best_f)

	return result
end

data = load_experimental_data("Parameter Estimation/ethane.csv")

res = estimate_xi_CH3!(model, data; lower = 0.0, upper = 1.0, seed = 42, max_iters = 1000)
println(res)