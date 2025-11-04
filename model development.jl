using LinearAlgebra
function bell_lot_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3" -0.015294175	-0.26365184	-0.933877108;
		       "CH2" -8.22E-05	-0.275395668	-0.2219383;
               "CH" 0.044865597	-0.390100245	0.714836414;
               "C"  0.19911712	0.220305695	1.292925164]

	groups = model.groups.groups[1]   
	num_groups = model.groups.n_groups[1]  # corresponding counts
	n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients
    S=model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10

    γ= 0.41722646
    
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
    end

	# Now sum across groups for each coefficient (column)
	n_g = sum(n_g_matrix, dims = 1)  # sums over rows for each column, result is 1x3 matrix

    m_gc = sum(model.groups.n_groups[1])
    R = Rgas()
    s_res = entropy_res(model, P, T, z)
    Z =  s_res / (R * m_gc)  # molar entropy term

    s_red = -s_res ./ R

    D = -0.1693155 * m_gc

    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z ^ 2 + D * Z ^ 3

    n_reduced = exp(ln_n_reduced) - 1.0

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

function bell_lot_viscosity_opt(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params::Dict)
    # === Extract optimization parameters ===
    n_alpha = params["n_alpha"]       # Dict("CH3" => (A, B, C), "CH2" => (...))
    γ       = params["gamma"]::Float64
    D_i       = params["D_i"]::Float64

    # === Model & group data ===
    groups      = model.groups.groups[1]
    num_groups  = model.groups.n_groups[1]
    S           = model.params.shapefactor
    σ           = diag(model.params.sigma.values) .* 1e10  # m → Å

    # === Volume contribution ===
    V = sum(num_groups .* S .* (σ .^ 3))

    # === Compute group contributions ===
    n_g_matrix = zeros(length(groups), 3)

    for (i, grp) in enumerate(groups)
        @assert haskey(n_alpha, grp) "Missing n_alpha entry for group '$grp'"
        Aα, Bα, Cα = n_alpha[grp]

        n_g_matrix[i, 1] = Aα * S[i] * σ[i]^3 * num_groups[i]
        n_g_matrix[i, 2] = Bα * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = Cα * num_groups[i]
    end

    n_g = vec(sum(n_g_matrix, dims = 1))
    m_gc = sum(num_groups)

    # === Thermodynamic terms ===
    R     = Rgas()
    s_res = entropy_res(model, P, T, z)
    z_term = s_res / (R * m_gc)
    s_red = -s_res / R

    # === Reduced viscosity correlation ===
    D = D_i * m_gc
    ln_n_reduced = n_g[1] + n_g[2]*z_term + n_g[3]*z_term^2 + D*z_term^3
    n_reduced = exp(ln_n_reduced) - 1.0

    # === Physical constants ===
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    Mw = Clapeyron.molecular_weight(model, z)
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
