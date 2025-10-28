function IB_pure_const(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; ξ = 1, C_i = 0.0 )
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.02848] # global parameters
	#ξ_pure = zeros(length(z))
	#n = model.groups.n_groups[1]

	ξ_mix = ξ #* (1-ξ_T*log(T))
    C = C_i
	#ξ_mix = ξ
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.75)) - C

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

function IB_pure_optimize(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; ξ = 1)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.0278307626] # global parameters
	#ξ_pure = zeros(length(z))
	#n = model.groups.n_groups[1]

	ξ_mix = ξ #* (1-ξ_T*log(T))
	#ξ_mix = ξ
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.75)) - 1

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

function IB_pure_optimize_T(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; ξ = 1, ξ_T = 1)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.0278307626] # global parameters
	#ξ_pure = zeros(length(z))
	#n = model.groups.n_groups[1]

	ξ_mix = ξ * (1-ξ_T*log(T))
	#ξ_mix = ξ
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.75)) - 1

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



function IB_viscosity_n_g_3(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; n_g_1 = 0.025, n_g_2 = 0.025,n_g_3 = 0.025)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	#n_g = [n_g_1, n_g_2, n_g_3] # global parameters
    n_g = [0.30136975, -0.11931025, 0.02531175]
	#ξ_pure = zeros(length(z))

	ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = [       "CH3" 0.40853;
			"CH2"  0.03833;
			"aCH"  0.18516;
			"cCH2" 0.103377]

		ξ_T = [       "CH3" 0.00;
			"CH2"  0.00;
			"aCH"  72.9645;
			"cCH2" -123.288]
        
        ξ = 0

		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)

			xi = ξ_i[ξ_i[:, 1].==groups[i], 2][1]
			xi_T = ξ_T[ξ_T[:, 1].==groups[i], 2][1]

			ξ = ξ + xi * (1 - xi_T / T) * num_groups[i]

			ξ_pure[j] = ξ
		end

	end
	ξ_mix = sum(z .* ξ_pure)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.75)) - 1

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


function IB_viscosity_global_2_6(model::EoSModel, P, T, z = StaticArrays.SA[1.0];
	ξ_i = Dict("CH3" => 0.4085265, "CH2" => 0.0383325),
	ξ_T = Dict("CH3" => 0.0, "CH2" => 0.0))
    #n_g_3 = 0.025)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, 0.02531175] # global parameters
	#ξ_pure = zeros(length(z))

	ξ = 0.0
	groups = model.groups.groups[1]
	num_groups = model.groups.n_groups[1]
	for i in 1:length(groups)
		g = groups[i]
		if !haskey(ξ_i, g)
			error("ξ_i missing entry for group \"$g\".")
		end
        T_b = saturation_temperature(model,P)
		ξ += ξ_i[g] * num_groups[i] * (1 - ξ_T[g] *(T/T_b[1]))
	end

	ξ_mix = sum(z .* ξ)
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



function IB_viscosity_TP(model::EoSModel, P, T, z = StaticArrays.SA[1.0];
	ξ_i = Dict("CH3" => 0.4085265, "CH2" => 0.0383325),
	#ξ_T = Dict("CH3" => 0.0, "CH2" => 0.0),
    n_g_3 = 0.028,
    C_i = Dict("CH3" => 0.0, "CH2" => 0.0))
    #n_g_3 = 0.025)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [0.30136975, -0.11931025, n_g_3] # global parameters
	#ξ_pure = zeros(length(z))

	ξ = 0.0
    C_T = 0.0
	groups = model.groups.groups[1]
	num_groups = model.groups.n_groups[1]
    #T_c = crit_pure(model)
	for i in 1:length(groups)
		g = groups[i]
		if !haskey(ξ_i, g)
			error("ξ_i missing entry for group \"$g\".")
		end
		C_T += C_i[g] * num_groups[i]
		ξ += ξ_i[g] * num_groups[i] #* (1 + ξ_T[g] *(T/T_c[1]))
	end

	ξ_mix = sum(z .* ξ)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.75)) - 1.0/C_T

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