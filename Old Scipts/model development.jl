using LinearAlgebra
function bell_lot_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.14544200778907007, 0.010683439643496976, -0.06148080798139255, 0.000157392327889979),
        "CH2" => (-1.1927784860733681, 0.011458247757004004, -0.008458302429596044,  0.00016724013588380074),
        "CH"  => (    -6.70322582565162, 0.013433432693828238, 0.03532440730533404, 0.0016759054663535538),
		"C"  => (   5.441006090748911, -0.0513546650750801, 0.1412593290192207, 0.0026320850844137205)
    )
    γ = 0.16499191915814165
 
    Mw = Clapeyron.molecular_weight(model, z)
 
    m0, m1, m2 = -1.019572741285313, 10.667125465466807, -7.956775819220652

    m = m0 + m1 * Mw + m2 * Mw ^ 2

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


function bell_lot_viscosity_mix(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
    n_alpha = Dict(
        "CH3" => ( 0.14544200778907007,  0.010683439643496976, -0.06148080798139255,  0.000157392327889979 ),
        "CH2" => (-1.1927784860733681,  0.011458247757004004, -0.008458302429596044,  0.00016724013588380074),
        "CH"  => (   -6.70322582565162,  0.013433432693828238,  0.03532440730533404,  0.0016759054663535538),
        "C"   => (    5.441006090748911, -0.0513546650750801,   0.1412593290192207,   0.0026320850844137205)
    )
    A_i = Dict(k => v[1] for (k,v) in n_alpha)
    γ = 0.16499191915814165

    components = model.groups.components
    tot_sf = zeros(length(z))
    V      = zeros(length(z))
    n_g    = zeros(length(z), 4)

    # Per-component m(Mw) polynomial (same as pure)
    m0, m1, m2 = -1.019572741285313, 10.667125465466807, -7.956775819220652

    for i in 1:length(z)
        comp_model = SAFTgammaMie([components[i]])
        groups     = comp_model.groups.groups[1]
        num_groups = comp_model.groups.n_groups[1]
        S          = comp_model.params.shapefactor
        σ          = diag(comp_model.params.sigma.values).* 1e10

        V[i] = sum(num_groups.* S.* (σ.^ 3))
        n_g_matrix = zeros(length(groups), 4)

        # per-component Mw, m, T_C (matches pure function algebra)
        Mw_i = Clapeyron.molecular_weight(comp_model)
        m_i  = m0 + m1 * Mw_i + m2 * Mw_i^2
        T_Ci = crit_pure(comp_model)[1]

        for j in 1:length(groups)
            group = groups[j]
            Aα, Bα, Cα, Dα = n_alpha[group]

            n_g_matrix[j,1] = 0.0                                          # will be set by A_vdw_mix later
            n_g_matrix[j,2] = Bα * S[j] * σ[j]^3 * num_groups[j] / (V[i]^γ)
            n_g_matrix[j,3] = Cα * num_groups[j] * (1 + m_i * sqrt(T / T_Ci))
            n_g_matrix[j,4] = Dα * S[j] * σ[j]^3 * num_groups[j]
        end

        n_g[i, :].= vec(sum(n_g_matrix, dims = 1))
        n_g[i,1] = A_vdw(comp_model,A_i)
        tot_sf[i] = sum(S.* num_groups)
    end

    # Mixture totals
    tot_sf_mix = sum(tot_sf.* z)
    #n_g_mix    = vec(sum(z.* n_g, dims = 1))
    n_g_mix    = vec(sum(z .* tot_sf./tot_sf_mix .* n_g,dims = 1))
    n_g_mix[1] = sum(n_g[:,1] .* z)


    # A-term: use your fixed mixer that collapses to pure value for one-hot z
    # (component-space vdW-like mixing)
    # Ensure you use the corrected version:
    #   A_vdw_mix(model, A_i, z) = z' * A_mat(z) * z, where A_mat is built from pure A_vdw per component
    #n_g_mix[1] = A_vdw_mix(model, A_i, z)

    # State functions
    R     = Clapeyron.Rgas()
    s_res = entropy_res(model, P, T, z)
    Z     = (-s_res) / (R * tot_sf_mix)    # same structure as pure (tot_sf for pure, tot_sf_mix here)

    # Reduced viscosity polynomial (same as pure model)
    ln_n_reduced = n_g_mix[1] + n_g_mix[2]*Z + n_g_mix[3]*Z^2 + n_g_mix[4]*Z^3
    n_reduced    = exp(ln_n_reduced) - 1.0

    # Number density, particle mass (mixture)
    N_A      = Clapeyron.N_A
    k_B      = Clapeyron.k_B
    ρ_molar  = molar_density(model, P, T, z)
    ρ_N      = ρ_molar * N_A
    Mw_mix   = Clapeyron.molecular_weight(model, z)
    m_part   = Mw_mix / N_A

    # Match pure algebra: no division by s_red^(2/3)
    n_res = (n_reduced) * (ρ_N^(2/3)) * sqrt(m_part * k_B * T)

    n_ideal = IB_CE_mix(model, T, z)
    viscosity = n_ideal + n_res
    return viscosity
end
bell_lot_viscosity_mix(model,1e5,300,[0.0,1.0])
model
bell_lot_test(SAFTgammaMie(["Octane"]),1e5,300)


function bell_lot_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3" -0.016248943	1.301165292	-13.21531378;
		       "CH2" 2.93E-04	1.011917658	-2.991386128;
               #"CH" 0.03930050901260097 -1.5984961240693298 10.093752915195292;
               "CH" 0.042101628	1.407726021	11.03083133
]
    # molar frac
    tau_i = ["CH3" 0.93985987;
		    "CH2" 0.605183564;
            #"CH" 0.5604734970596769;
            "CH" 0.612296858127202]
    #vdw, arithmetic mean
    #tau_i = ["CH3" 0.823964850820917;
#		    "CH2" 0.581851010454827;
#            "CH" 0.805233975276969]

	groups = model.groups.groups[1]   
	num_groups = model.groups.n_groups[1]  # corresponding counts
	n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients
    tau = zeros(length(groups))

    S=model.params.shapefactor.values
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

function bell_lot_viscosity_sf(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3"  -0.015087401852988442 1.1793288696973334 -10.799687755027858;
		       "CH2"  -0.0006393202735293168 0.9103543410756307 -1.9017722602334455;
               #"CH" 0.03930050901260097 -1.5984961240693298 10.093752915195292;
               "CH"  0.04611757489106076 2.064388453805704	7.452487373056837]
    #n_alpha = Dict(
    #        "CH3" => (-0.016318734957479455, 1.2247150298572662 -10.78613820563853),
    #        "CH2" => (2.93E-04,	-1.011917658,	-2.991386128),
    #        "CH"  => (0.042101628,	-1.407726021,	11.03083133)
    #        )
    # molar frac
    tau_i = Dict(
            "CH3" => (1.1714896072382803),
            "CH2" => (1.2),
            "CH"  => (1.6275002221121222)
        )
    #vdw, arithmetic mean
    #tau_i = ["CH3" 0.823964850820917;
#		    "CH2" 0.581851010454827;
#            "CH" 0.805233975276969]

	groups = model.groups.groups[1]   
	num_groups = model.groups.n_groups[1]  # corresponding counts
	n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients

    S=model.params.shapefactor.values
    σ = diag(model.params.sigma.values) .* 1e10

    γ= 0.45
    D_i = 12.536185542576174

    V= sum(num_groups.*S.*(σ.^3))

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
    end

	# Now sum across groups for each coefficient (column)
	n_g = sum(n_g_matrix, dims = 1)  # sums over rows for each column, result is 1x3 matrix

    m_gc = sum(model.groups.n_groups[1].*S) # total number of groups

    tau_mix = tau_OFE(model,tau_i) # makes it mole fraction based
   
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
    S           = model.params.shapefactor.values
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
    m_gc = sum(num_groups)

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    #z_term = (s_res) / (R * m_gc * log(T)^ tau_mix)

    s_red = -s_res / R

    z_term = (-s_res / (s_id) +  log(-s_res / R) / m_gc)


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

function bell_lot_viscosity_opt_s_idref(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # === Extract optimization parameters ===
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    γ       = params["gamma"]::Float64
    D_i     = params["D_i"]

    # === Model & group data ===
    groups      = model.groups.groups[1]
    num_groups  = model.groups.n_groups[1]
    S           = model.params.shapefactor.values
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

    total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])
    m_gc = sum(model.groups.n_groups[1])
    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    #z_term = (s_res) / (R * m_gc * log(T)^ tau_mix)

    s_red = -s_res / R
    s_id = entropy_ideal(model, P, T, z)

    z_term = (-s_res / (s_id) +  log(-s_res / R) / total_sf)
    #z_term = (-s_res / (R * m_gc)) # +  log(-s_res / R) / total_sf)


    Mw = Clapeyron.molecular_weight(model, z)

    # === Reduced viscosity correlation ===
    #D = (D_1+D_2/Mw)^(-1) #* m_gc
    D = D_i * total_sf
    ln_n_reduced = n_g[1] + n_g[2]*z_term + n_g[3]*z_term^2 + D*z_term^3
    n_reduced = exp(ln_n_reduced) - 1.0

    # === Physical constants ===
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    
    m = Mw / N_A

    # === Residual contribution ===
    n_res = n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T) / (s_red ^ (2/3))

    # === Chapman–Enskog or mixture viscosity ===
    viscosity = if length(z) == 1
        IB_CE(model, T) + n_res
    else
        IB_CE_mix(model, T, z) + n_res
    end

    return viscosity
end



function bell_lot_viscosity_opt_ogref(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # === Extract optimization parameters ===
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    γ       = params["gamma"]::Float64
    tau_i   = params["tau_i"]
    D_i     = params["D_i"]

    # === Model & group data ===
    groups      = model.groups.groups[1]
    num_groups  = model.groups.n_groups[1]
    S           = model.params.shapefactor.values
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
    #m_gc = sum(num_groups)
    m_gc = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

    #tau_mix = tau_OFE(model,tau_i) # makes it mole fraction based
    tau_mix = sum(tau)./sum(num_groups)

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    #z_term = (s_res) / (R * m_gc * log(T)^ tau_mix)
    z_term =  (-s_res) / (R * log(T)^ tau_mix)
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
    n_res = n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T)

    # === Chapman–Enskog or mixture viscosity ===
    viscosity = if length(z) == 1
        IB_CE(model, T) + n_res
    else
        IB_CE_mix(model, T, z) + n_res
    end

    return viscosity
end

function IB_pure_const_6param(
    model::EoSModel,
    P, T,
    z = StaticArrays.SA[1.0];
    ξ::Float64 = 1.0,
    C_i::Float64 = 0.0,
    a1::Float64 = 59.22750961,
    a2::Float64 = -0.30570183,
    b1::Float64 = -58.83680498,
    b2::Float64 = 0.31422912,
    c::Float64  = -0.01913978,
    d::Float64  = 8.76828433
)
    """
    Overall viscosity using a 6-parameter sigmoid correlation for ln(n_red + 1),
    consistent with your dimensionless collapse.

    Inputs:
      - model, P, T, z: Clapeyron model and state (pure by default: z = SA[1.0])
      - ξ: scaling factor for the entropy coordinate x (defaults to 1.0)
      - C_i: small offset subtracted from n_red after exp(log) - 1
      - (a1, a2, b1, b2, c, d): 6-parameter coefficients (defaults from your fit)

    Returns:
      - viscosity (same units as IB_CE / IB_CE_mix for the given model)
    """

    # Thermodynamic entropies and critical quantities
    s_res = entropy_res(model, P, T, z)
    s_id  = Clapeyron.entropy(model, P, T, z) - s_res

    Tc, Pc = crit_pure(model)
    s_crit = entropy_res(model, Pc, Tc)

    # Reduced entropy coordinate (scaled by ξ)
    x_es   = (-s_res / s_id) + log(s_res / s_crit)
    x_scaled = x_es / ξ

    # 6-parameter model for ln(n_red + 1)
    ln_n_plus = ((a1 + a2 * s_res) / (1 + exp(c * x_scaled)) +
                 (b1 + b2 * s_res) / (1 + exp(-c * x_scaled))) * x_scaled +
                d / s_crit

    # Dimensionless reduced viscosity (subtract optional offset C_i)
    n_reduced = exp(ln_n_plus) - 1.0 - C_i

    # Number density and mass per molecule
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N     = ρ_molar * N_A

    Mw = Clapeyron.molecular_weight(model, z)
    m  = Mw / N_A

    # Avoid potential division-by-zero in the (x_scaled)^(2/3) factor
    # (the theoretical form uses x^(2/3); here we ensure numerical robustness)
    denom = (abs(x_scaled))^(2/3)

    # Dimensional residual viscosity
    n_res = (n_reduced * (ρ_N^(2/3)) * sqrt(m * k_B * T)) / denom

    # Baseline viscosity (pure vs mixture)
    viscosity = IB_CE_mix(model, T, z) + n_res

    return viscosity
end