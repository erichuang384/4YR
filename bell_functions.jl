include("util_functions.jl")

function IB_viscosity_pure(model::EoSModel, P, T, ξ, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.02531175] # global parameters

	ξ_mix = ξ
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

function IB_viscosity_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [4.931216347208786, -4.380361669951116, 1.5692739255277655] # global parameters
	ξ_pure = zeros(length(z))
	#A =  7.809602262149181
	#B = -0.34096357773369207

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 2.0986652810214323;
			"CH2"  0.18253549764390836;
			"aCH"  0.88788;
			"cCH2" 0.762199846/6]
        ξ = 0

		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)

			xi = ξ_i[ξ_i[:, 1].==groups[i], 2][1]

			ξ +=  + xi* num_groups[i] #* (1 - xi_T * log(T)) 

			ξ_pure[j] = ξ
		end

	end
	#T_c = crit_pure(model)
	ξ_mix = sum(z .* ξ_pure)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	ln_n_reduced = (n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8))
	n_reduced = exp(ln_n_reduced) - 1
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


function IB_viscosity_const(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.028487736198671] # global parameters
	ξ_pure = zeros(length(z))
	C_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 0.419265814680057;
			"CH2" 0.0374958277215246;
			"aCH"  0.88788;
			"cCH2" 0.774571519/6]

		C_i = ["CH3" -0.028548540202457;
			"CH2"  0.500255137911703;
			"aCH"  0.0;
			"cCH2" 1.5045939/6]
        
        ξ = 0
		C = 0
		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)

			xi = ξ_i[ξ_i[:, 1].==groups[i], 2][1]
			#xi_T = ξ_T[ξ_T[:, 1].==groups[i], 2][1]
			c_group = C_i[C_i[:, 1].==groups[i], 2][1]

			ξ = ξ + xi  * num_groups[i] #* (1 - xi_T * log(T))
			C = C + c_group * num_groups[i]

			ξ_pure[j] = ξ
			C_pure[j] = C
		end

	end
	#T_c = crit_pure(model)
	ξ_mix = sum(z .* ξ_pure)
	C_mix = sum(z .* C_pure)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.75)) - 1/C_mix

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


function IB_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.02531175] # global parameters
	#n_g = [0.7918253, -0.4792172, 0.06225174]
	n_exp = [ 1.76, 2.134, 2.608]
	ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 0.4085265;
			"CH2"  0.0383325;
			"aCH"  0.142683;
			"cCH2" 0.7614/5]
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

	#n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8)) - 1
	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ n_exp[1] + n_g[2] .* (s_red ./ ξ_mix) .^ n_exp[2] + n_g[3] .* (s_red ./ ξ_mix) .^ n_exp[3] .- 0.6) - 1

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

function IB_viscosity_4param(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell
	"""
	n_g = [-0.448046, 1.012681, -0.381869, 0.054674] # global parameters
	ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 0.484458;
			"CH2"  0.047793;
			"aCH"  0.174573333]
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
