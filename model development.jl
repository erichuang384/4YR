using LinearAlgebra
function bell_lot_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3" -0.016248943	1.301165292	-13.21531378;
		       "CH2" 2.93E-04	1.011917658	-2.991386128;
               "CH" 0.03930050901260097 -1.5984961240693298 10.093752915195292;
               "C"  0.19911712	0.220305695	1.292925164]
    # molar frac
    tau_i = ["CH3" 0.93985987;
		    "CH2" 0.605183564;
            "CH" 0.5604734970596769]
    #vdw, arithmetic mean
    #tau_i = ["CH3" 0.823964850820917;
#		    "CH2" 0.581851010454827;
#            "CH" 0.805233975276969]

	groups = model.groups.groups[1]   
	num_groups = model.groups.n_groups[1]  # corresponding counts
	n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients
    tau = zeros(length(groups))

    S=model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10

    γ= 0.437793675
    D_i = 7.176085783

    V= sum(num_groups.*S.*(σ.^3))
    #B=sum((n_groups.*)./(V.^γ))

    # A calculate
    for i in 1:length(groups)
    # find group index in n_i
        row = findfirst(x -> x == groups[i], n_alpha[:,1])
        A_alpha = n_alpha[row, 2]
        B_alpha = n_alpha[row, 3]
        C_alpha = n_alpha[row, 4]

        n_g_matrix[i,1] = A_alpha * S[i] * σ[i] ^ 3 * num_groups[i]
        n_g_matrix[i,2] = B_alpha * S[i] * σ[i] ^ 3 * num_groups[i] / (V^γ)
        n_g_matrix[i,3] = C_alpha * num_groups[i]

        row_tau = findfirst(x -> x == groups[i], tau_i[:,1])
        tau[i] = tau_i[row_tau, 2] * num_groups[i]
    end

	# Now sum across groups for each coefficient (column)
	n_g = sum(n_g_matrix, dims = 1)  # sums over rows for each column, result is 1x3 matrix

    m_gc = sum(model.groups.n_groups[1]) # total number of groups

    tau_mix = sum(tau) ./ m_gc # makes it mole fraction based
   
    R = Rgas()
    s_res = entropy_res(model, P, T, z)
    Z =  (-s_res) / (R * m_gc * log(T)^ tau_mix)  # molar entropy term

    s_red = -s_res ./ R

    Mw = Clapeyron.molecular_weight(model, z)

    D = D_i * m_gc

    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z ^ 2 + D * Z ^ 3

    n_reduced = exp(ln_n_reduced) - 1.0

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

	m = Mw / N_A

	n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))

	if length(z) == 1
		viscosity = IB_CE(model, T) + n_res
	else
		viscosity = IB_CE_mix(model, T, z) + n_res
	end

	return viscosity
end

function bell_lot_viscosity_opt(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
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
    m_gc = sum(num_groups)

    tau_mix = tau_OFE(model,tau_i) # makes it mole fraction based

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    #z_term = (s_res) / (R * m_gc * log(T)^ tau_mix)
    z_term = (s_res) / (R * log(T)^ tau_mix)
    s_red = -s_res / R

    Mw = Clapeyron.molecular_weight(model, z)

    # === Reduced viscosity correlation ===
    #D = (D_1+D_2/Mw)^(-1) #* m_gc
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



function bell_lot_viscosity_opt_noD(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # === Extract optimization parameters ===
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    tau_i = params["tau_i"]

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

    ln_n_reduced = n_g[1] + n_g[2]*z_term + n_g[3]*z_term^2
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

function bell_lot_viscosity_noD(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3" -0.00839281621531174 -0.2536455462605472 -1.2079684228897345;
		       "CH2" 0.00081612760670802	-0.554009241078019	0.4562494998529377]
    # molar frac
    tau_i = ["CH3" 0.24074197784699647;
		    "CH2" 0.5527120103735335]
    #vdw, arithmetic mean
    #tau_i = ["CH3" 0.823964850820917;
#		    "CH2" 0.581851010454827;
#            "CH" 0.805233975276969]

	groups = model.groups.groups[1]   
	num_groups = model.groups.n_groups[1]  # corresponding counts
	n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients
    tau = zeros(length(groups))

    S=model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10

    γ= 0.45

    V= sum(num_groups.*S.*(σ.^3))
    #B=sum((n_groups.*)./(V.^γ))

    # A calculate
    for i in 1:length(groups)
    # find group index in n_i
        row = findfirst(x -> x == groups[i], n_alpha[:,1])
        A_alpha = n_alpha[row, 2]
        B_alpha = n_alpha[row, 3]
        C_alpha = n_alpha[row, 4]

        n_g_matrix[i,1] = A_alpha * S[i] * σ[i] ^ 3 * num_groups[i]
        n_g_matrix[i,2] = B_alpha * S[i] * σ[i] ^ 3 * num_groups[i] / (V^γ)
        n_g_matrix[i,3] = C_alpha * num_groups[i]

        row_tau = findfirst(x -> x == groups[i], tau_i[:,1])
        tau[i] = tau_i[row_tau, 2] * num_groups[i]
    end

	# Now sum across groups for each coefficient (column)
	n_g = sum(n_g_matrix, dims = 1)  # sums over rows for each column, result is 1x3 matrix

    m_gc = sum(model.groups.n_groups[1]) # total number of groups

    tau_mix = sum(tau) ./ m_gc # makes it mole fraction based
   
    R = Rgas()
    s_res = entropy_res(model, P, T, z)
    Z =  (s_res) / (R * m_gc * log(T)^ tau_mix)  # molar entropy term

    s_red = -s_res ./ R

    Mw = Clapeyron.molecular_weight(model, z)

    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z ^ 2

    n_reduced = exp(ln_n_reduced) - 1.0

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

	m = Mw / N_A

	n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))

	if length(z) == 1
		viscosity = IB_CE(model, T) + n_res
	else
		viscosity = IB_CE_mix(model, T, z) + n_res
	end

	return viscosity
end


function bell_lot_viscosity_opt_epsilon(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
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
    tau = zeros(length(groups))

    for (i, grp) in enumerate(groups)
        @assert haskey(n_alpha, grp) "Missing n_alpha entry for group '$grp'"
        Aα, Bα, Cα = n_alpha[grp]
        tau_dum = tau_i[grp]

        n_g_matrix[i, 1] = Aα * S[i] * σ[i]^3 * num_groups[i]
        n_g_matrix[i, 2] = Bα * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = Cα * num_groups[i]
        tau[i] = tau_dum * num_groups[i]
    end

    n_g = vec(sum(n_g_matrix, dims = 1))
    m_gc = sum(num_groups)

    tau_mix = sum(tau)
    epsilon =  ϵ_OFE(model)

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    z_term = (s_res) / (R * m_gc * log(T /(tau_mix * epsilon)))
    s_red = -s_res / R

    Mw = Clapeyron.molecular_weight(model, z)

    # === Reduced viscosity correlation ===
    #D = (D_1+D_2/Mw)^(-1) #* m_gc
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

    tau_mix = tau_OFE(model,tau_i) # makes it mole fraction based

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
    n_reduced = exp(ln_n_reduced) - 1.0

    # === Physical constants ===
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    
    m = Mw / N_A

    # === Residual contribution ===
    n_res = n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T) / (z_term^(2/3))

    # === Chapman–Enskog or mixture viscosity ===
    viscosity = if length(z) == 1
        IB_CE(model, T) + n_res
    else
        IB_CE_mix(model, T, z) + n_res
    end

    return viscosity
end