include("util_functions.jl")

function IB_viscosity_3param_T(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
    """
    Overall Viscosity using method proposed by Ian Bell, 3 parameters
    """
    n_g = [0.30136975, -0.11931025, 0.02531175] # global parameters
    ξ_pure = zeros(length(z))
 
    for j ∈ 1:length(z)
 
        ξ_i = ["CH3" 0.40853;
        "CH2"  0.03833;
        "aCH"  0.18516;
        "cCH2" 0.103377]
    
        ξ_T = ["CH3" 0.00;
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

            ξ = ξ + xi * (1-xi_T/T) * num_groups[i]
 
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