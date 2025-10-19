include("util_functions.jl")

function IB_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell
	"""
	n_g = [-0.448046, 1.012681, -0.381869, 0.054674] # global parameters
	ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 0.484458;
			"CH2"  0.047793;
			"aCH"  0.17447]
		ξ = 0
		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)
			value = ξ_i[ξ_i[:, 1].==groups[i], 2][1]
			ξ = ξ + value * num_groups[i]

			ξ_pure[j] = ξ
		end

	end

	ξ_mix = sum(z .* ξ_pure)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) + n_g[2] .* (s_red ./ ξ_mix) .^ (1.5) + n_g[3] .* (s_red ./ ξ_mix) .^ (2) + n_g[4] .* (s_red ./ ξ_mix) .^ (2.5)) - 1

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

function IB_viscosity_3param(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.301491667, -0.143158667, 0.036576] # global parameters
	ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 0.484458;
			"CH2"  0.047793;
			"aCH"  0.143064667]
		ξ = 0
		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)
			value = ξ_i[ξ_i[:, 1].==groups[i], 2][1]
			ξ = ξ + value * num_groups[i]

			ξ_pure[j] = ξ
		end

	end

	ξ_mix = sum(z .* ξ_pure)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8)) - 1

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

function IB_CE_mix(model::EoSModel, T, z)
	# need to define models
	components = model.groups.components
	models = []

	for i in 1:length(components)
		push!(models, SAFTgammaMie([components[i]]))
	end
	# calculate CE pure
	viscosity_CE = IB_CE.(models[:], T)
	Mw = Clapeyron.molecular_weight.(models[:])

	#Φ matrix
	n = length(viscosity_CE)
	phi = zeros(n, n)

	for i in 1:n, j in 1:n
		phi[i, j] = (1 + sqrt(viscosity_CE[i] / viscosity_CE[j]) * (Mw[j] / Mw[i])^(0.25))^2 /
					sqrt(8 * (1 + Mw[i] / Mw[j]))
	end

	phi_ij = [sum(z[j] * phi[i, j] for j in 1:n) for i in 1:n]

	visc_mix = sum(z[i] * viscosity_CE[i] / phi_ij[i] for i in 1:n)
	return visc_mix
end

function IB_CE(model::EoSModel, T)
	"""
	Chapman-Enskog Theory for two component because way sigma specified
	Replace with one component something
	"""
	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B
	σ = σ_OFE(model)
	Mw = Clapeyron.molecular_weight(model) # in kg/mol
	#m_gc = sum(model.groups.n_groups[1])
	Ω = Ω⃰(model, T)
	visc = 5 / 16 * sqrt(Mw * k_B * T / (N_A * pi)) / (σ^2 * Ω)
	return visc
end


