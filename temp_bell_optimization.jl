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

function IB_viscosity_TP(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; 
	ξ_i = Dict("CH3" => 0.4, "CH2" => 0.3),
	#ξ_T = Dict("CH3" => 0.4, "CH2" => 0.3),
    n_g_1 = 0.2, n_g_2 = -0.4, n_g_3 = 0.02,
	tau_i = Dict("CH3" => 0.0, "CH2" => 0.0),
	exp_1 = 1.8, exp_2 = 2.4, exp_3 = 2.8)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [n_g_1, n_g_2, n_g_3] # global parameters
	exp_i = [exp_1, exp_2,exp_3]
	
	#ξ_pure = zeros(length(z))

	ξ = 0.0
	tau = 0.0

	groups = model.groups.groups[1]
	num_groups = model.groups.n_groups[1]
	S = model.params.shapefactor.values
    #T_c = crit_pure(model)
	for i in 1:length(groups)
		g = groups[i]
		if !haskey(ξ_i, g)
			error("ξ_i missing entry for group \"$g\".")
		end
		#C_T += C_i[g] * num_groups[i]
		ξ += ξ_i[g] * num_groups[i]
		tau += tau_i[g] * num_groups[i]
	end

	tot_sf = sum(num_groups .* S)
	tau = tau /sum(num_groups)
	ξ_mix = sum(z .* ξ)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	
	s_red = tot_sf *  ((-s_res ./ R) ^ tau ) ./ log(T)

	#ln_n_reduced = (n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8))
	#running in vscode
	#ln_n_reduced = (n_g[1] .* (s_red ./ ξ_mix) + n_g[2] .* (s_red ./ ξ_mix) .^ (1.5) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.0) + n_g[4] .* (s_red ./ ξ_mix) .^ (2.5))

	ln_n_reduced = (n_g[1] .* (s_red ./ ξ_mix) .^ (exp_i[1]) + n_g[2] .* (s_red ./ ξ_mix) .^ (exp_i[2]) + n_g[3] .* (s_red ./ ξ_mix) .^ (exp_i[3]))

	n_reduced = exp.(ln_n_reduced) .- 1.0
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

function IB_viscosity_newref(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; 
	ξ_i = Dict("CH3" => 0.499189, "CH2" => 0.002368),
	#ξ_T = Dict("CH3" => 0.4, "CH2" => 0.3),
    n_g_1 = 1.604172, n_g_2 = -1.941327, n_g_3 =  1.070956)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [n_g_1, n_g_2, n_g_3] # global parameters

	ξ = 0.0

	groups = model.groups.groups[1]
	num_groups = model.groups.n_groups[1]
	S = model.params.shapefactor.values

	for i in 1:length(groups)
		g = groups[i]
		if !haskey(ξ_i, g)
			error("ξ_i missing entry for group \"$g\".")
		end
		ξ += ξ_i[g] * num_groups[i]
	end

	tot_sf = sum(num_groups .* S)

	ξ_mix =  sum(z .* ξ)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	
	s_id = entropy_ideal(model, P, T, z)
	s_crit = entropy_res_crit(model)
	s_red = (-s_res/s_id + log(s_res/s_crit))

	ln_n_reduced = (n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8))

	n_reduced = exp.(ln_n_reduced) .- 1.0
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

function IB_viscosity_newref_pure_xi(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; 
	ξ = 1.0)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g =  [1.604172, -1.941327, 1.070956]# global parameters
	exp_i = [1.8, 2.4, 2.8]
	
	s_res = entropy_res(model, P, T, z)
	
	s_id = entropy_ideal(model, P, T, z)
	s_crit = entropy_res_crit(model)
	s_red = (-s_res/s_id + log(s_res/s_crit))

	ln_n_reduced = (n_g[1] .* (s_red ./ ξ) .^ (exp_i[1]) + n_g[2] .* (s_red ./ ξ) .^ (exp_i[2]) + n_g[3] .* (s_red ./ ξ) .^ (exp_i[3]))

	n_reduced = exp.(ln_n_reduced) .- 1.0
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


function IB_viscosity_newref_vdw(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; 
	ξ_i = Dict("CH3" => 0.4, "CH2" => 0.3),
	#ξ_T = Dict("CH3" => 0.4, "CH2" => 0.3),
    n_g_1 = 0.2, n_g_2 = -0.4, n_g_3 = 0.02)
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_g = [n_g_1, n_g_2, n_g_3] # global parameters

	ξ_mix =  xi_OFE(model, ξ_i)

	s_res = entropy_res(model, P, T, z)
	
	s_id = entropy_ideal(model, P, T, z)
	s_crit = entropy_res_crit(model)
	s_red = (-s_res/s_id + log(s_res/s_crit))

	ln_n_reduced = (n_g[1] .* (s_red ./ ξ_mix) .^ (1.8) + n_g[2] .* (s_red ./ ξ_mix) .^ (2.4) + n_g[3] .* (s_red ./ ξ_mix) .^ (2.8))

	n_reduced = exp.(ln_n_reduced) .- 1.0
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