function bell_lot_viscosity_mix(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
    n_alpha = Dict(
        "CH3" => [-0.016248943, 1.301165292, -13.21531378],
        "CH2" => [2.93e-4, 1.011917658, -2.991386128],
        "CH"  => [0.042101628, 1.407726021, 11.03083133],
        "C" => [0.24921570099125565 -7.899465014177413 16.815242578194816]
    )
    tau_i = Dict(
        "CH3" => 0.93985987,
        "CH2" => 0.605183564,
        "CH"  => 0.612296858127202,
        "C" => -0.6337686308472408
    )

    γ = 0.437793675
    D_i = 7.176085783

    components = model.groups.components
    m = zeros(length(z))
    V = zeros(length(z))
    n_g = zeros(length(z), 3)
    tau_mix_terms = zeros(length(z))

    for i in 1:length(z)
        comp_model = SAFTgammaMie([components[i]])
        groups = comp_model.groups.groups[1]
        num_groups = comp_model.groups.n_groups[1]
        S = comp_model.params.shapefactor
        σ = diag(comp_model.params.sigma.values) .* 1e10

        V[i] = sum(num_groups .* S .* (σ .^ 3))
        n_g_matrix = zeros(length(groups), 3)
        tau_comp = zeros(length(groups))

        for j in 1:length(groups)
            group = groups[j]
            A_alpha, B_alpha, C_alpha = n_alpha[group]
            n_g_matrix[j,1] = A_alpha * S[j] * σ[j]^3 * num_groups[j]
            n_g_matrix[j,2] = B_alpha * S[j] * σ[j]^3 * num_groups[j] / (V[i]^γ)
            n_g_matrix[j,3] = C_alpha * num_groups[j]
            tau_comp[j] = tau_i[group] * num_groups[j]
        end

        n_g[i, :] .= vec(sum(n_g_matrix, dims = 1))
        m[i] = sum(num_groups)
        tau_mix_terms[i] = sum(tau_comp)
    end

    m_mean = sum(m .* z)
    tau_mix = sum(tau_mix_terms .* z) / m_mean

    R = Rgas()
    s_res = entropy_res(model, P, T, z)
    s_red = -s_res / R
    Z = (-s_res) / (R * m_mean * (log(T)^tau_mix))

    n_g_mix = vec(sum(z .* n_g, dims = 1))
    D = D_i * m_mean

    ln_n_reduced = n_g_mix[1] + n_g_mix[2]*Z + n_g_mix[3]*Z^2 + D*Z^3
    n_reduced = exp(ln_n_reduced) - 1.0

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar * N_A
    Mw = Clapeyron.molecular_weight(model, z)
    m_particle = Mw / N_A

    n_res = (n_reduced * (ρ_N^(2/3)) * sqrt(m_particle * k_B * T)) / (s_red^(2/3))

    viscosity = IB_CE_mix(model, T, z) + n_res
    return viscosity
end

#=
function Grunburg_viscosity_mixture(model::EoSModel,P,T,z)
    """
    Mixing rule for binary mixture
    """
    components = model.groups.components
    #models = []

    #Δ_i = [-0.1, 0.096, 0.204, 0.433]
    
    n = zeros(length(z))
    #N_i = zeros(length(z))
    for i = 1:length(z)
        model_i=SAFTgammaMie([components[i]])
        n[i]=bell_lot_viscosity_good(model_i,P,T)

        #Δ[i] = sum(model_i.groups.n_groups[1].*Δ_i) 
        #N_i[i] = sum(model_i.groups.n_groups[1])
    end
    
    #W = (0.3161.*(N_i[1] - N_i[2]).^2)./(sum(N_i))
    #G = Δ[1] - Δ[2] + W
    #G=0
    #n_mix = exp(sum(z .* log.(n)) + G)
    #n_mix = exp(z[1] * log(n[1]) + z[2] * log(n[2]) + z[1] * z[2] * G)
    
    n_mix = exp(z[1] .* log(n[1]) .+ z[2] .* log(n[2]))

    return n_mix
end
=#


function Grunburg_viscosity_mixture(model::EoSModel,P,T,z)
    """
    Mixing rule for binary mixture
    """
    components = model.groups.components
    
    
    n = zeros(length(z))
    ρ = zeros(length(z))

    for i = 1:length(z)
        model_i=SAFTgammaMie([components[i]])
        n[i]=bell_lot_viscosity_mix(model_i,P,T)
        ρ[i]= mass_density(model_i,P,T)
    end
    
    ρ_mix = mass_density(model,P,T,z)



    n_mix = exp(z[1] * log(n[1]) + z[2] * log(n[2]) + z[1] * z[2] * G)

    return n_mix
end





models = [
    SAFTgammaMie(["heptamethylnonane","dodecane"]),
    SAFTgammaMie(["2,2,4-trimethylpentane","dodecane"]),
    SAFTgammaMie(["2,2,4-trimethylpentane","octane"]),
    SAFTgammaMie(["dodecane","heptadecane"]),
    SAFTgammaMie(["eicosane" ,"heptane"]),
    #SAFTgammaMie(["hexane" ,"dodecane"]),
    #SAFTgammaMie(["pentane","octane"]),
    SAFTgammaMie(["Squalane" ,"octane"]),
    SAFTgammaMie(["squalane" ,"hexane"]),
    SAFTgammaMie(["tridecane" , "heptamethylnonane"]),
    SAFTgammaMie(["tridecane","pentadecane"])
]

data_paths = [
    "Daanyaal Mixture/2,2,4,4,6,8,8-heptamethylnonane - dodecane.csv",
    "Daanyaal Mixture/2,2,4-trimethylpentane - dodecane.csv",
    "Daanyaal Mixture/2,2,4-trimethylpentane - octane.csv",
    "Daanyaal Mixture/dodecane - heptadecane.csv",
    "Daanyaal Mixture/eicosane - heptane.csv",
    #"Experimental DATA/Mixtures/hexane - dodecane.csv",
    #"Experimental DATA/Mixtures/pentane - octane.csv",
    "Daanyaal Mixture/Squalane - octane.csv",
    "Daanyaal Mixture/squalane- hexane.csv",
    "Daanyaal Mixture/tridecane - 2,2,4,4,6,8,8-heptamethylnonane.csv",
    "Daanyaal Mixture/tridecane - pentadecane.csv"
]


datasets = [load_experimental_data(p) for p in data_paths]

AAD_lot = zeros(length(models))
AAD_grun = zeros(length(models))

for j in 1:length(models) 
    data = datasets[j]
    P=data[:,1]
    T=data[:,2]
    x1=data[:,4]
    n_exp=data[:,3]

    #n_mix_lot = zeros(length(T))
    n_mix_grun = zeros(length(T))
    for i in 1:length(T)
        z = [x1[i], 1-x1[i]]
        #n_mix_lot[i] = bell_lot_viscosity_mix(models[j],P[i],T[i],z)
        n_mix_grun[i] = Grunburg_viscosity_mixture(models[j],P[i],T[i],z)
        
    end

    #AAD_lot[j] = sum(abs.( (n_exp .- n_mix_lot)./n_exp))/length(n_exp)
    AAD_grun[j] = sum(abs.( (n_exp .- n_mix_grun)./n_exp))/length(n_exp)
end
print(AAD_grun)