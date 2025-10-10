using Clapeyron
using Plots
using LinearAlgebra
using CSV
using DataFrames

include("all_functions.jl")

#model = SAFTgammaMie(["hexane","heptane","octane"])
#model = SAFTgammaMie(["methane","butane"])
function CE_mix(model::EoSModel,T,z)
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


model = SAFTgammaMie(["hexane","heptane"])
z = [0.64, 1-0.64]

N=100
P_range=LinRange(1e5,715e5,N)
T=303
P_bar=P_range./1e5

#n_reduced=zeros(N)
#n_CE=zeros(N)
viscosity_mix=zeros(N)
for i in 1:N
    vicosity_reduced_mix = reduced_viscosity_mixture(model,P_range[i],T,z)
    viscosity_CE_mix = CE_mix(model,T,z) 
    viscosity_mix[i] = vicosity_reduced_mix.*viscosity_CE_mix
end


# Plot viscosity vs. Pressure


plot(
    P_bar, viscosity_mix,
    xlabel = L"P\ \mathrm{/Bar}",
    ylabel = L"\eta\ \mathrm{/(Pa\cdot s)}",
    title  = "Viscosity at $(T) K",
    lw = 2,
    label = "Mixture",   # label for legend
    grid = false,
    guidefont = font(16)
)




viscosity_hex=zeros(N)
for i in 1:N
    vicosity_reduced_hex= reduced_viscosity(model_hex,P_range[i],T)
    viscosity_CE_hex = dilute_gas_viscosity(model_hex,T) 
    viscosity_hex[i]=vicosity_reduced_hex.*viscosity_CE_hex
end

viscosity_hep=zeros(N)
for i in 1:N
    vicosity_reduced_hep = reduced_viscosity(model_hep,P_range[i],T)
    viscosity_CE_hep = dilute_gas_viscosity(model_hep,T) 
    viscosity_hep[i] = vicosity_reduced_hep.*viscosity_CE_hep
end

# Plot viscosity vs. Pressure


plot!(
    P_bar, viscosity_hep,
    xlabel = L"P\ \mathrm{/Bar}",
    ylabel = L"\eta\ \mathrm{/(Pa\cdot s)}",
    title  = "Viscosity at $(T) K",
    lw = 2,
    label = "Heptane",   # label for legend
    grid = false,
    guidefont = font(16)
)

plot!(
    P_bar, viscosity_hex,
    lw = 2,
    label = "Hexane",    # label for second curve
)



####

    df = CSV.read("Hex_Hept Mix 303K x=0.4.csv", DataFrame)


P = df[:, 2]
n_experimental = df[:, 4] 


# Add experimental data as points
plot!(P./1e5, n_experimental;
      seriestype = :scatter,    # points instead of lines
      label = "Experimental x_hex=0.64",
      color = "green"    
)  




