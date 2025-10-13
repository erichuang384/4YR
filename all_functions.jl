function σ_OFE(model::EoSModel)
    """
    sigma pure fluid equivalent, Papaioannou 
    """
    xₛₖ = x_sk(model)
    σ = cbrt(xₛₖ' * (model.params.sigma).^3 * xₛₖ)
    return σ
end

function ϵ_OFE(model::EoSModel)
    """
    epsilon pure fluid equivalent, Papaioannou 
    """
    xₛₖ = x_sk(model)
    ϵ_kl = model.params.epsilon.values
    ϵ = xₛₖ' * ϵ_kl * xₛₖ
    return ϵ
end

function x_sk(model::EoSModel)
    """
    x_sk, Papaioannou 
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
    reduced viscosity from Lotgering 2015
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
    Chapman-Enskog Viscosity for Mixture by Wilke 1950 
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
    """
    Method for viscosity of mixtures by Lotgering 2018
    """
    components = model.groups.components
    models = []
 
    for i in 1:length(components)
        push!(models,SAFTgammaMie([components[i]]))
    end

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

function λ_r_OFE(model::EoSModel)
    """
    λ_r pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    λ_kl = model.params.lambda_r.values
    λ = xₛₖ' * λ_kl * xₛₖ
    return λ
end
    # Compute a_i^(2,2)(λr) using Eq. (S.4)
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