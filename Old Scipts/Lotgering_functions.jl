include("util_functions.jl")

function Lotgering_viscosity(model::EoSModel,P,T)
    """
    Lotgering method for pure fluid
    """
    vicosity_reduced = Lotgering_reduced_viscosity(model,P,T)
    viscosity_CE = Lotgering_dilute_gas_viscosity(model,T) 
    n = vicosity_reduced.*viscosity_CE 
    return n
end

function Lotgering_viscosity_mix(model::EoSModel,P,T,z)
    """
    Viscosity of mixture from Lotgering 2018 paper
    """
    vicosity_reduced_mix = Lotgering_reduced_viscosity_mixture(model,P,T,z)
    viscosity_CE_mix = Lotgering_CE_mix(model,T,z) 
    n = vicosity_reduced_mix.*viscosity_CE_mix 
    return n
end

function Lotgering_reduced_viscosity(model::EoSModel,P,T)
    """
    Lotgering Dimensionless Viscosity for pure
    """
    n_α= model.groups.n_groups[1] 
    S=model.params.shapefactor
    σ = diag(model.params.sigma.values)
    a_α=[1.1e28,-1.92e27];b_α=[-4.32e15,-7.51e15]  #paramters from 2017 slides 
    #a_α=[1.77e27,4.83e27];b_α=[-2.25e15,-9.01e15]   #paramters from 2017 paper for n-alkanes, for liquid region
    γ=0.45 #Constant for n-alkanes
    
    V=(sum(n_α.*S.*(σ.^3)))
    A=sum(n_α.*S.*(σ.^3).*a_α)
    B=sum((n_α.*S.*(σ.^3).*b_α)./(V.^γ))
    
    s_res=entropy_res(model,P,T)
    m_gc = sum(model.groups.n_groups[1])
    k_B = 1.380649e-23
    R=8.314 #Using R instead of kB
    z=(s_res./(R.*m_gc))  # molar entropy

    n_reduced=exp(A+B.*z)
    return n_reduced
end

function Lotgering_reduced_viscosity_mixture(model::EoSModel,P,T,z)
    """
    Lotgering Viscosity for mixture
    """
    components = model.groups.components

    a_α=[1.1e28,-1.92e27];b_α=[-4.32e15,-7.51e15]  #paramters from 2017 slides 
    #a_α=[1.77e27,4.83e27];b_α=[-2.25e15,-9.01e15]   #paramters from 2017 paper for n-alkanes, for liquid region
    γ=0.45 #Constant for n-alkanes

    s_res=entropy_res(model,P,T,z) 
    m=zeros(length(z))
    V=zeros(length(z))
    A=zeros(length(z))
    B=zeros(length(z))

    for i = 1:length(z)
        model=SAFTgammaMie([components[i]])
        m[i] = (sum(model.groups.n_groups[1]))
        n_α= model.groups.n_groups[1] 
        S=model.params.shapefactor
        σ = diag(model.params.sigma.values)
        V[i]=(sum(n_α.*S.*(σ.^3)))
        A[i]=sum(n_α.*S.*(σ.^3).*a_α)
        B[i]=sum((n_α.*S.*(σ.^3).*b_α)./(V[i].^γ))
    end
    m_mean=sum(m.*z)
    #k_B = 1.380649e-23
    R=8.314 
    s_reduced=(s_res./(R.*m_mean))  # molar entropy
    n_reduced_mix = exp(sum(z.*A) + sum(((z.*m)./m_mean).*B.*s_reduced))
    return n_reduced_mix
end

function Lotgering_CE_mix(model::EoSModel,T,z)
    # need to define models
    components = model.groups.components
    models = []
 
    for i in 1:length(components)
        push!(models,SAFTgammaMie([components[i]]))
    end
    # calculate CE pure
    viscosity_CE = Lotgering_dilute_gas_viscosity.(models[:],T)
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

function Lotgering_dilute_gas_viscosity(model::EoSModel,T)
    """
    Chapman-Enskog Theory for two component because way sigma specified
    Replace with one component something
    """
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    σ = σ_OFE(model)
    Mw = Clapeyron.molecular_weight(model) # in kg/mol
    m_gc = sum(model.groups.n_groups[1])
    Ω = Ω⃰(model,T)
    visc = 5/16*sqrt(Mw*k_B*T/(m_gc*N_A*pi))/(σ^2*Ω)
    return visc
end



function bell_lot_viscosity_mix(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3" -0.016248943	1.301165292	-13.21531378;
		       "CH2" 2.93E-04	1.011917658	-2.991386128;
               "CH" 0.042101628	1.407726021	11.03083133]
    # molar frac
    tau_i = ["CH3" 0.93985987;
		    "CH2" 0.605183564;
            "CH" 0.612296858127202]
    γ= 0.437793675
    D_i = 7.176085783
    n_g_matrix_mix = zeros(length(z),3)
    m_gc_mix = zeros(length(z))
    tau_mixture = zeros(length(z))
    D_matrix = zeros(length(z))
    components = model.groups.components
# ==================
    for j in 1:length(z)
        comp_model = SAFTgammaMie([components[j]])
	    groups = comp_model.groups.groups[1]   
	    num_groups = comp_model.groups.n_groups[1]  # corresponding 
        S = comp_model.params.shapefactor
        σ = diag(comp_model.params.sigma.values) .* 1e10

	    n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients
        tau = zeros(length(groups))

        V = sum(num_groups.*S.*(σ.^3))

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

        m_gc = sum(comp_model.groups.n_groups[1]) # total number of groups

        tau_mix = sum(tau) ./ m_gc # makes it mole fraction 
        
        n_g_matrix_mix[j,:] = n_g
        m_gc_mix[j] = m_gc
        tau_mixture[j] = tau_mix
        
        D = D_i * m_gc
        D_matrix[j] = D
    end
    # ===================================
    m_mean = sum(m_gc_mix .* z)
    tau_overall = sum(tau_mixture .* z .* m_gc_mix ./ m_mean)

    n_g_1_overall = 0
    n_g_2_overall = 0
    n_g_3_overall = 0
    n_g_4_overall = 0

    R = Rgas()

    for i in 1:length(z)
        n_g_1_overall += sum(n_g_matrix_mix[i,1] .* z[i])
        n_g_2_overall += sum(n_g_matrix_mix[i,2] .* z[i] .* m_gc_mix[i] ./ m_mean)
        n_g_3_overall += sum(n_g_matrix_mix[i,3] .* z[i] .* m_gc_mix[i] ./ m_mean)
        n_g_4_overall += sum(D_matrix[i] .* z[i] .* m_gc_mix[i] ./ m_mean)
    end
    s_res = entropy_res(model, P, T, z)
    s_red = (-s_res/R)
    Z = (-s_res) / (R * m_mean * log(T)^ tau_overall)
    ln_n_reduced = n_g_1_overall + n_g_2_overall * Z + n_g_3_overall * Z ^ 2 + n_g_4_overall * Z ^ 3

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


function lot_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell, 3 parameters
	"""
	n_alpha = ["CH3" -0.0116953315864751 0.4223154028105611 -1.1486176828290624;
		       "CH2" -0.0016439802421435658 0.3174580677512573 -0.22561637761145772]

    γ= 0.45148295117527165
    D_i = 0.184353816497909
    n_g_matrix_mix = zeros(length(z),3)
    m_gc_mix = zeros(length(z))

    D_matrix = zeros(length(z))
    components = model.groups.components
# ==================
    for j in 1:length(z)
        comp_model = SAFTgammaMie([components[j]])
	    groups = comp_model.groups.groups[1]   
	    num_groups = comp_model.groups.n_groups[1]  # corresponding 
        S = comp_model.params.shapefactor
        σ = diag(comp_model.params.sigma.values) .* 1e10

	    n_g_matrix = zeros(length(groups), 3)  # rows: groups, cols: 3 coefficients

        V = sum(num_groups.*S.*(σ.^3))

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

        m_gc = sum(comp_model.groups.n_groups[1]) # total number of groups

        
        n_g_matrix_mix[j,:] = n_g
        m_gc_mix[j] = m_gc
        
        D = D_i * m_gc
        D_matrix[j] = D
    end
    # ===================================
    m_mean = sum(m_gc_mix .* z)

    n_g_1_overall = 0
    n_g_2_overall = 0
    n_g_3_overall = 0
    n_g_4_overall = 0

    R = Rgas()

    for i in 1:length(z)
        n_g_1_overall += sum(n_g_matrix_mix[i,1] .* z[i])
        n_g_2_overall += sum(n_g_matrix_mix[i,2] .* z[i] .* m_gc_mix[i] ./ m_mean)
        n_g_3_overall += sum(n_g_matrix_mix[i,3] .* z[i] .* m_gc_mix[i] ./ m_mean)
        n_g_4_overall += sum(D_matrix[i] .* z[i] .* m_gc_mix[i] ./ m_mean)
    end
    s_res = entropy_res(model, P, T, z)
    s_red = (-s_res/R)
    Z = (-s_res) / (R * m_mean)
    ln_n_reduced = n_g_1_overall + n_g_2_overall * Z + n_g_3_overall * Z ^ 2 + n_g_4_overall * Z ^ 3

    n_reduced = exp(ln_n_reduced)

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

    Mw = Clapeyron.molecular_weight(model, z)

	m = Mw / N_A

	#n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))
    n_res = n_reduced .* IB_CE_mix(model, T, z) 
    viscosity = n_res

#	if length(z) == 1
#		viscosity = IB_CE(model, T) + n_res
#	else
#		viscosity = IB_CE_mix(model, T, z) + n_res
#	end

	return viscosity
end