include("util_functions.jl")

function Ω⃰(model::EoSModel, T)
    """
    Collision Integral correlation by Fokin et al.
    """
    λ_r = λ_r_OFE(model)
    a_vals =  a²²ᵢ(λ_r)
    ϵ = ϵ_OFE(model)
    T⃰ = T / ϵ

    ln_Omega = -2/λ_r * log(T⃰) + log(1 - 2/(3*λ_r)) + sum(a_vals[i] * (1/T⃰)^((i-1)/2) for i in 1:6)

    return exp(ln_Omega)
end

function Lotgering_viscosity(model::EoSModel,P,T)
    """
    Overall viscosity from Lotgering method
    """
    vicosity_reduced = Lotgering_reduced_viscosity(model,P,T)
    viscosity_CE = Lotgering_dilute_gas_viscosity(model,T) 
    n = vicosity_reduced.*viscosity_CE 
    return n
end

function IB_dilute_gas_viscosity(model::EoSModel,T)
    """
    Chapman-Enskog Theory for two component because way sigma specified
    Replace with one component something
    """
    N_A = 6.0221408e23
    k_B = 1.380649e-23
    σ = σ_OFE(model)
    Mw = Clapeyron.molecular_weight(model) # in kg/mol
    #m_gc = sum(model.groups.n_groups[1])
    Ω = Ω⃰(model,T)
    visc = 5/16*sqrt(Mw*k_B*T/(N_A*pi))/(σ^2*Ω)
    return visc
end


function IB_viscosity(model::EoSModel,P,T)
    """
    Overall Viscosity uisng method proposed by Ian Bell 
    ξ can be changed / predicted   
    """
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674]
    ξ_hexane = 1.1650526 #SAJED
    #ξ_hexane = 1.06863 #PC SAFT
    #ξ_hexane = 1.15994 #LKP

    R=8.314
    s_res=entropy_res(model,P,T)
    s_red=-s_res./R

    #n_reduced= exp(n_g[1].*(s_red./ξ_hexane) + n_g[2].*(s_red./ξ_hexane).^(1.5) + n_g[3].*(s_red./ξ_hexane).^(2) + n_g[4].*(s_red./ξ_hexane).^(2.5)) -1
    n_reduced= exp(n_g[1].*(s_red./ξ) + n_g[2].*(s_red./ξ).^(1.5) + n_g[3].*(s_red./ξ).^(2) + n_g[4].*(s_red./ξ).^(2.5)) -1

    N_A = 6.0221408e23
    k_B = 1.380649e-23

    ρ_molar=molar_density(model,P,T)
    ρ_N = ρ_molar.*N_A

    Mw = Clapeyron.molecular_weight(model)
    m=Mw/N_A

    n_res= (n_reduced.*(ρ_N.^(2/3)).*sqrt.(m.*k_B.*T))./((s_red).^(2/3))

    viscosity = IB_dilute_gas_viscosity(model,T) + n_res
    return viscosity
end

function Lotgering_CE_mix(model::EoSModel,T,z)
    # need to define models
    components = model.groups.components
    models = []
 
    for i in 1:length(components)
        push!(models,SAFTgammaMie([components[i]]))
    end
    # calculate CE pure
    viscosity_CE = dilute_gas_viscosity.(models[:],T)
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



function Lotgering_reduced_viscosity_mixture(model::EoSModel,P,T,z)
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

function Lotgering_viscosity_mix(model::EoSModel,P,T,z)
    """
    Viscosity of mixture from Lotgering 2018 paper
    """
    vicosity_reduced_mix = Lotgering_reduced_viscosity_mixture(model,P,T,z)
    viscosity_CE_mix = Lotgering_CE_mix(model,T,z) 
    n = vicosity_reduced_mix.*viscosity_CE_mix 

    return n
end