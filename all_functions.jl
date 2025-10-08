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

    n_α= model.groups.n_groups[1] 
    S=model.params.shapefactor
    σ = diag(model.params.sigma.values)
    a_α=[1.1e28,-1.92e27];b_α=[-4.32e15,-7.51e15]  #paramters from 2017 paper for n-alkanes, for liquid region
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