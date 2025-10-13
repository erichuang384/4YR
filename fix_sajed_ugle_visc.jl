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