using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays
include("bell_functions.jl")
include("model development.jl")
#include("lotgering_functions.jl")

models = [
	SAFTgammaMie(["Butane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Pentane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Hexane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Heptane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Octane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Nonane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Decane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Undecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Dodecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Tridecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Tetradecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Pentadecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Hexadecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["Heptadecane"], idealmodel = WalkerIdeal),
	SAFTgammaMie(["n-eicosane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
#    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
#    SAFTgammaMie(["9-octylheptadecane"]),
    SAFTgammaMie(["squalane"])
]


#model_pentadecane = SAFTgammaMie(["pentadecane"])

#models = [model_pentane, model_hexane, model_heptane, model_octane, model_nonane, model_decane]

labels = ["Butane", "Pentane", "Hexane", "Heptane", "Octane", "Nonane", "Decane", "Undecane", "Dodecane", "Tridecane", "Tetradecane",
	"Pentadecane",
	"Hexadecane", "Heptadecane", "n-eicosane"]

#experimental values
exp_nonane = CSV.read("Validation Data/Nonane DETHERM.csv", DataFrame)
exp_undecane = CSV.read("Validation Data/Undecane DETHERM.csv", DataFrame)
exp_tridecane = CSV.read("Validation Data/Tridecane DETHERM.csv", DataFrame)
exp_pentadecane = CSV.read("Validation Data/Pentadecane DETHERM.csv", DataFrame)
exp_heptadecane = CSV.read("Validation Data/Heptadecane DETHERM.csv", DataFrame)

exp_butane = CSV.read("Training Data/Butane DETHERM.csv", DataFrame)
exp_pentane = CSV.read("Training Data/Pentane DETHERM.csv", DataFrame)
exp_hexane = CSV.read("Training Data/Hexane DETHERM.csv", DataFrame)
exp_heptane = CSV.read("Training Data/Heptane DETHERM.csv", DataFrame)
exp_octane = CSV.read("Training Data/Octane DETHERM.csv", DataFrame)
exp_decane = CSV.read("Training Data/Decane DETHERM.csv", DataFrame)
exp_dodecane = CSV.read("Training Data/Dodecane DETHERM.csv", DataFrame)
exp_tetradecane = CSV.read("Training Data/Tetradecane DETHERM.csv", DataFrame)
exp_hexadecane = CSV.read("Training Data/Hexadecane DETHERM.csv", DataFrame)
exp_eicosane = CSV.read("Training DATA/n-eicosane.csv", DataFrame)

exp_2_6_10_14_tetramethylpentadecane = CSV.read("Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",DataFrame)
#exp_2_methylpropane     = CSV.read("Training DATA/Branched Alkane/2-methylpropane.csv",DataFrame)
exp_2_methylbutane      = CSV.read("Training DATA/Branched Alkane/2-methylbutane.csv",DataFrame)
exp_2_methylpentane     = CSV.read("Training DATA/Branched Alkane/2-methylpentane.csv",DataFrame)
#exp_9_octylheptadecane  = CSV.read("Training DATA/Branched Alkane/9-octylheptadecane.csv",DataFrame)
exp_squalane            = CSV.read("Training DATA/Branched Alkane/squalane.csv",DataFrame)


exp_data = [exp_butane, exp_pentane, exp_hexane, exp_heptane, exp_octane, exp_nonane, exp_decane, exp_undecane, exp_dodecane, exp_tridecane, exp_tetradecane,
	exp_pentadecane,
	exp_hexadecane, exp_heptadecane, exp_eicosane,
    exp_2_6_10_14_tetramethylpentadecane, exp_2_methylpropane, exp_2_methylbutane, exp_2_methylpentane, 
#    exp_9_octylheptadecane, 
    exp_squalane]

AAD = zeros(length(models))

#AAD_pentadecane = zeros(length(exp_pentadecane[:,1]))
#index = [2, 3, 5, 7, 9, 10, 12, 14]
for i in 1:length(models)
	T_exp = exp_data[i][:, 2]
	n_exp = exp_data[i][:, 3]
	P_exp = exp_data[i][:, 1]
	n_calc = bell_lot_test.(models[i], P_exp, T_exp)

	AAD[i] = sum(abs.((n_exp .- n_calc) ./ n_exp)) / length(P_exp)
end
println("AAD = ", AAD)
mean(AAD)
mean(AAD[2:end-1])


function lot_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
	# --- Default hard-coded parameters (used if params not provided) ---
	#    default_n_alpha = Dict(
	#        "CH3" => (-0.0116953315864751, 0.4223154028105611, -1.1486176828290624),
	#        "CH2" => (-0.0016439802421435658, 0.3174580677512573, -0.22561637761145772)
	#    )
	default_n_alpha = Dict(
		"CH3" => (-0.0152941746, -0.26365184, -0.933877108),
		"CH2" => (-0.0000822187, -0.275395668, -0.2219383),
	)
	default_gamma = 0.41722646
	default_D_i = -0.1693155

	# --- Use optimized / CSV parameters if passed ---
	n_alpha = params !== nothing && haskey(params, "n_alpha") ? params["n_alpha"] : default_n_alpha
	γ       = params !== nothing && haskey(params, "gamma") ? params["gamma"] : default_gamma
	D_i     = params !== nothing && haskey(params, "D_i") ? params["D_i"] : default_D_i

	nz = length(z)
	n_g_matrix_mix = zeros(nz, 3)
	m_gc_mix = zeros(nz)
	D_matrix = zeros(nz)

	components = model.groups.components

	# --- Loop through each component ---
	for j in 1:nz
		comp_model = SAFTgammaMie([components[j]])
		groups = comp_model.groups.groups[1]
		num_groups = comp_model.groups.n_groups[1]
		S = comp_model.params.shapefactor
		σ = diag(comp_model.params.sigma.values) .* 1e10

		n_g_matrix = zeros(length(groups), 3)
		V = sum(num_groups .* S .* (σ .^ 3))

		for i in 1:length(groups)
			gname = groups[i]
			if !haskey(n_alpha, gname)
				error("Group $gname not found in parameter dictionary")
			end
			A, B, C = n_alpha[gname]

			n_g_matrix[i, 1] = A * S[i] * σ[i]^3 * num_groups[i]
			n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
			n_g_matrix[i, 3] = C * num_groups[i]
		end

		n_g = sum(n_g_matrix, dims = 1)
		m_gc = sum(num_groups)

		n_g_matrix_mix[j, :] = vec(n_g)
		m_gc_mix[j] = m_gc
		D_matrix[j] = D_i * m_gc
	end

	m_mean = sum(m_gc_mix .* z)

	n_g_1_overall = 0.0
	n_g_2_overall = 0.0
	n_g_3_overall = 0.0
	n_g_4_overall = 0.0

	R = Clapeyron.Rgas()

	for i in 1:nz
		n_g_1_overall += n_g_matrix_mix[i, 1] * z[i]
		n_g_2_overall += n_g_matrix_mix[i, 2] * z[i] * m_gc_mix[i] / m_mean
		n_g_3_overall += n_g_matrix_mix[i, 3] * z[i] * m_gc_mix[i] / m_mean
		n_g_4_overall += D_matrix[i] * z[i] * m_gc_mix[i] / m_mean
	end

	s_res = entropy_res(model, P, T, z)

	s_red = -s_res / R

	Z = (s_res) / (R * m_mean)

	ln_n_reduced = n_g_1_overall + n_g_2_overall * Z + n_g_3_overall * Z^2 + n_g_4_overall * Z^3

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model, z)

	m = Mw / N_A
	n_reduced = exp(ln_n_reduced) - 1.0

	n_res = (n_reduced) .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T) ./ ((s_red) .^ (2 / 3))

	viscosity = n_res + IB_CE_mix(model, T, z)
	return viscosity
end

function b_lot_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
	# --- Default hard-coded parameters (used if params not provided) ---
	#    default_n_alpha = Dict(
	#        "CH3" => (-0.0116953315864751, 0.4223154028105611, -1.1486176828290624),
	#        "CH2" => (-0.0016439802421435658, 0.3174580677512573, -0.22561637761145772)
	#    )
	default_n_alpha = Dict(
		"CH3" => (0.22311992203010386, 0.01892763056088764, -0.09968670672385233 ),
		"CH2" => ( -1.357797583343831, 0.031874217787294, -0.001683690848038567),
        "CH"  => (0.3175076261986114, 0.00023421863834581987, 0.00014377419901436968)
	)
	default_gamma = 0.3175076261986114
	D_CH3 = 0.00023421863834581987
	D_CH2 = 0.00014377419901436968
    m0, m1, m2 = -0.9927325409136291, 7.98070549196157, 7.09279125158157

    Mw = Clapeyron.molecular_weight(model, z)

    m = m0 + m1 * Mw + m2 + Mw ^ 2

	# --- Use optimized / CSV parameters if passed ---
	n_alpha = params !== nothing && haskey(params, "n_alpha") ? params["n_alpha"] : default_n_alpha
	γ       = params !== nothing && haskey(params, "gamma") ? params["gamma"] : default_gamma
	D_i     = params !== nothing && haskey(params, "D_i") ? params["D_i"] : default_D_i

	nz = length(z)
	n_g_matrix_mix = zeros(nz, 3)
	m_gc_mix = zeros(nz)
	tot_sf_mix = zeros(nz)
	D_matrix = zeros(nz)

	x_sk_model = x_sk(model)

	components = model.groups.components

	# --- Loop through each component ---
	for j in 1:nz
		comp_model = SAFTgammaMie([components[j]])
		groups = comp_model.groups.groups[1]
		num_groups = comp_model.groups.n_groups[1]
		S = comp_model.params.shapefactor
		σ = diag(comp_model.params.sigma.values) .* 1e10


		n_g_matrix = zeros(length(groups), 3)
		V = sum(num_groups .* S .* (σ .^ 3))


		for i in 1:length(groups)
			gname = groups[i]
			if !haskey(n_alpha, gname)
				error("Group $gname not found in parameter dictionary")
			end
			A, B, C = n_alpha[gname]

			n_g_matrix[i, 1] = A * S[i] * σ[i]^3 * num_groups[i]
			n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
			n_g_matrix[i, 3] = C * num_groups[i] * 1
		end

		n_g = sum(n_g_matrix, dims = 1)
		m_gc = sum(num_groups)
		tot_sf = sum(S .* num_groups)

		n_g_matrix_mix[j, :] = vec(n_g)
		tot_sf_mix[j] = tot_sf
		m_gc_mix[j] = m_gc
		#D_matrix[j] = D_i * m_gc
		D_matrix[j] = D_i * tot_sf
	end

	m_mean = sum(m_gc_mix .* z)
	sf_mean = sum(tot_sf_mix .* z)

	n_g_1_overall = 0.0
	n_g_2_overall = 0.0
	n_g_3_overall = 0.0
	n_g_4_overall = 0.0

	R = Clapeyron.Rgas()

	for i in 1:nz
		n_g_1_overall += n_g_matrix_mix[i, 1] * z[i]
		n_g_2_overall += n_g_matrix_mix[i, 2] * z[i] * m_gc_mix[i] / m_mean
		n_g_3_overall += n_g_matrix_mix[i, 3] * z[i] * m_gc_mix[i] / m_mean
		n_g_4_overall += D_matrix[i] * z[i] * m_gc_mix[i] / m_mean
	end


	s_res = entropy_res(model, P, T, z)

	s_red = -s_res / R

	Z = (-s_res) / (R * sf_mean)

	ln_n_reduced = n_g_1_overall + n_g_2_overall * Z + n_g_3_overall * Z^2 + n_g_4_overall * Z^3

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A


	m = Mw / N_A
	n_reduced = exp(ln_n_reduced) - 1.0

	n_res = (n_reduced) .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T) ./ ((s_red) .^ (2 / 3))

	viscosity = n_res + IB_CE_mix(model, T, z)
	return viscosity
end


function bell_lot_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
	# --- Default hard-coded parameters (used if params not provided) ---
	#    default_n_alpha = Dict(
	#        "CH3" => (-0.0116953315864751, 0.4223154028105611, -1.1486176828290624),
	#        "CH2" => (-0.0016439802421435658, 0.3174580677512573, -0.22561637761145772)
	#    )
	n_alpha = Dict(
		"CH3" => (0.22311992203010386, 0.01892763056088764, -0.09968670672385233, 0.00023421863834581987),
		"CH2" => (-1.357797583343831, 0.031874217787294, -0.001683690848038567, 0.00014377419901436968),
        "CH"  => (-5.345367204169419, 0.044309711230071595, 0.0951614852547662, 0.0013834292725805246)
	)
	γ = 0.3175076261986114

    Mw = Clapeyron.molecular_weight(model, z)

    m0, m1, m2 =  -0.9927325409136291, 7.98070549196157, 7.09279125158157

    m = m0 + m1 * Mw + m2 * Mw ^ 2

	x_sk_model = x_sk(model)
    
    A_i = Dict(k => v[1] for (k,v) in n_alpha)

	groups = model.groups.groups[1]
	num_groups = model.groups.n_groups[1]
	S = model.params.shapefactor
	σ = diag(model.params.sigma.values) .* 1e10


	n_g_matrix = zeros(length(groups), 4)
	V = sum(num_groups .* S .* (σ .^ 3))

    crit_point = crit_pure(model)
    T_C = crit_point[1]

	for i in 1:length(groups)
		gname = groups[i]
		if !haskey(n_alpha, gname)
			error("Group $gname not found in parameter dictionary")
		end
		A, B, C, D = n_alpha[gname]

		n_g_matrix[i, 1] = 0
		n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
		n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*sqrt(T/T_C))
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
	end

	n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)

	tot_sf = sum(S .* num_groups)

	R = Clapeyron.Rgas()

	s_res = entropy_res(model, P, T, z)

	Z = (-s_res) / (R * tot_sf)

	ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model, z)

	m = Mw / N_A
	n_reduced = exp(ln_n_reduced) - 1.0

	n_res = (n_reduced) .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)# ./ ((s_red) .^ (2 / 3))

	viscosity = n_res + IB_CE(model, T)
	return viscosity
end




#=
p_all = []  # store plots

for i in 1:length(models)
	model = models[i]
	data = exp_data[i]

	P_exp = data[:,1]
	T_exp = data[:,2]
	n_exp = data[:,3]

	n_calc = bell_lot_viscosity.(model, P_exp, T_exp)
	AAD_plot = ((n_exp .- n_calc) ./ n_exp) .* 100

	res_ent = entropy_res.(model, P_exp, T_exp) ./ (-Rgas())

	p = scatter(
		res_ent, AAD_plot,
		xlabel = L"s^+",
		ylabel = L"AD (\%)",
		title = "AD% vs Residual Entropy for $(labels[i])",
		legend = false,
		markersize = 5,
		color = :blue
	)

	push!(p_all, p)
end
p_all[1]
p_all[2]
p_all[3]
p_all[4]
p_all[5]
p_all[6]
p_all[7]
p_all[8]
p_all[9]
p_all[10]
p_all[11]
p_all[12]
p_all[13]
p_all[14]
=#
#savefig(p_all[1],"AAD vs res ent Octane")
#savefig(p_all[5],"AD vs res ent Dodecane")
