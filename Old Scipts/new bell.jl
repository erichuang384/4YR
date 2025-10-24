include("util_functions.jl")


function IB_viscosity_new(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""

	n_i = [    "CH3" 0.199166 12.1976 0.0723016;
		"CH2" -0.0115388 -62.7495 134.238]

	groups = model.groups.groups[1]   # e.g. ["CH3", "CH2"]
	num_groups = model.groups.n_groups[1]  # corresponding counts
    x_groups = num_groups./ sum(num_groups)

	n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients
    #n_g
    S=model.params.shapefactor
    σ = diag(model.params.sigma.values)

    γ= [1.0536, 1.43602] 
    
    V=(sum(num_groups.*S.*(σ.^3))) * 1e30
    #B=sum((n_groups.*)./(V.^γ))

    # A calculate
    for i in 1:length(groups)
    # find group index in n_i
        row = findfirst(x -> x == groups[i], n_i[:,1])
        n_g_matrix[i,1] = n_i[row, 2] * num_groups[i]
    end

    # B calculate
    for i in 1:length(groups)
        row = findfirst(x -> x == groups[i], n_i[:,1])
        value = n_i[row,3]
        n_g_matrix[i,2] = value * x_groups[i] / (V^γ[1])
    end

    # C calculate
    for i in 1:length(groups)
        row_i = findfirst(x -> x == groups[i], n_i[:,1])
        a_i = n_i[row_i, 4]
        term = 0.0
        for j in 1:length(groups)
            row_j = findfirst(x -> x == groups[j], n_i[:,1])
            a_j = n_i[row_j, 4]
            term += x_groups[i] * x_groups[j] * sqrt(a_i * a_j)
        end
        n_g_matrix[i,3] = term / (V^γ[2])
    end


	# Now sum across groups for each coefficient (column)
	n_g = sum(n_g_matrix, dims = 1)  # sums over rows for each column, result is 1x3 matrix
	
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red) .^ (1.8) + n_g[2] .* (s_red) .^ (2.4) + n_g[3] .* (s_red) .^ (2.8)) - 1

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


