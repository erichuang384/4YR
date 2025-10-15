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

function λ_r_OFE(model::EoSModel)
    """
    λ_r pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    λ_kl = model.params.lambda_r.values
    λ = xₛₖ' * λ_kl * xₛₖ
    return λ
end

function  a²²ᵢ(λ_r)
    """
    Matrix of coefficients for Collision integral by Fokin et al.
    """

    a²² = [0.0           0.113086e1    0.234799e2    0.310127e1;
    0.0           0.551559e1   -0.137023e3    0.185848e2;
    0.325909e-1  -0.292925e2    0.243741e3    0.0;
    0.697682      0.590192e2   -0.143670e3   -0.123518e3;
   -0.564238     -0.430549e2    0.0           0.137282e3;
    0.126508      0.104273e2    0.150601e2   -0.408911e2]
    a = zeros(6)
    for i in 1:6
        a[i] = a²²[i,1] + a²²[i,2]/λ_r + a²²[i,3]/λ_r^2 + a²²[i,4]/λ_r^3
    end
    return a
end


function Lotgering_dilute_gas_viscosity(model::EoSModel,T)
    """
    Chapman-Enskog Theory for two component because way sigma specified
    Replace with one component something
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

function Lotgering_reduced_viscosity(model::EoSModel,P,T)
    """
    Dimensionelss Viscosity
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

function Ω⃰_LJ(model::EoSModel,T)
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