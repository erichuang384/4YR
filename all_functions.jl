function σ_OFE(model::EoSModel)
    """
    sigma pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    σ = cbrt(xₛₖ' * (model.params.sigma).^3 * xₛₖ)
    return σ
end

function ϵ_OFE(model::EoSModel)
    """
    epsilon pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    ϵ_kl = model.params.epsilon.values
    ϵ = xₛₖ' * ϵ_kl * xₛₖ
    return ϵ
end

function x_sk(model::EoSModel)
    """
    x_sk
    """
    s = model.params.shapefactor
    segment = model.params.segment
    gc = 1:length(model.groups.flattenedgroups)
    comps = 1:length(model.groups.components)
    v = float.(model.groups.n_groups)
    for k in 1:length(gc)
        for i in 1:length(comps)
            v[i][k] = v[i][k]*s[k]*segment[k]
        end
    end
    xₛₖ = v./sum(v[1])
    return xₛₖ[1]
end

function Ω⃰(model::EoSModel,T)
    """
    Reduced Collision Integral
    For two component, need to work on getting epsilon in general
    """
    
    ϵ = ϵ_OFE(model)
    T⃰ = T / ϵ
    if !(0.3 < T⃰ < 100)
        print("Unviable T for correlation")
    end
    #think that k_B is already in epsilon parameter according to sample calc
    Ω = 1.16145*(T⃰)^(-0.14874)+0.52487*exp(-0.77320*(T⃰))+2.16178*exp(-2.43787*(T⃰))
    return Ω
end

function dilute_gas_viscosity(model::EoSModel,T)
    """
    Chapman-Enskog Theory
    """
    N_A = 6.0221408e23
    k_B = 1.380649e-23
    σ = σ_OFE(model)
    Mw = Clapeyron.molecular_weight(model) # in kg/mol
    m_gc = sum(model.groups.n_groups[1])
    Ω = Ω⃰(model,T)
    visc = 5/16*sqrt(Mw*k_B*T/(m_gc*N_A*pi))/(σ^2*Ω)
    return visc
end

function reduced_viscosity(model::EoSModel,P,T)
    """
    reduced viscosity from Lotgering
    """

    n_α= model.groups.n_groups[1] 
    S=model.params.shapefactor
    σ = diag(model.params.sigma.values)
    a_α=[1.1e28,-1.92e27];
    b_α=[-4.32e15,-7.51e15]  #paramters from 2017 paper for n-alkanes, for liquid region
    γ=0.45 #Constant for n-alkanes
    

    V=sum(n_α.*S.*(σ.^3))
    A=sum(n_α.*S.*(σ.^3).*a_α)
    B=sum((n_α.*S.*(σ.^3).*b_α)./(V.^γ))
    
    s_res=entropy_res(model,P,T)
    m_gc = sum(model.groups.n_groups[1])
    k_B = 1.380649e-23
    R = 8.31446261815324 #Using R instead of kB
    z=(s_res./(R.*m_gc))  # molar entropy

    n_reduced=exp(A+B.*z)
    return n_reduced
end

function CE_mix(model::EoSModel,T,z)
    """
    Chapman-Enskog Viscosity for Mixture
    """
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
end


function reduced_viscosity_mixture(model::EoSModel,P,T,z)
    components = model.groups.components
    models = []
 
    for i in 1:length(components)
        push!(models,SAFTgammaMie([components[i]]))
    end

    # calculate CE pure
    #viscosity_CE = dilute_gas_viscosity.(models[:],T)

    a_α=[1.1e28,-1.92e27];b_α=[-4.32e15,-7.51e15]  #paramters from 2017 paper for n-alkanes, for liquid region
    γ=0.45 #Constant for n-alkanes
  
    s_res=entropy_res(model,P,T,z) 

    m=zeros(length(z))
    V=zeros(length(z))
    A=zeros(length(z))
    B=zeros(length(z))

    for i = 1:length(z)
        m[i] = (sum(model.groups.n_groups[i])).*z[i]

        n_α= model.groups.n_groups[i] 
        S=model.params.shapefactor
        σ = diag(model.params.sigma.values)

        V[i]=(sum(n_α.*S.*(σ.^3)))
        A[i]=sum(n_α.*S.*(σ.^3).*a_α)
        B[i]=sum((n_α.*S.*(σ.^3).*b_α)./(V[i].^γ))
    end
    m_mean=sum(m)

    #k_B = 1.380649e-23
    R=8.314 
    s_reduced=(s_res./(R.*m_mean))  # molar entropy

    n_reduced_mix = sum(z.*A) + sum(((z.*m)./m_mean).*B.*s_reduced)
    return n_reduced_mix
end