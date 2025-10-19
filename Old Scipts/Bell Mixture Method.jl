using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings

include("bell_functions.jl")
model = SAFTgammaMie(["hexane","heptane"])


function IB_viscosity_mix(model::EoSModel,P,T,z)
    """
    Overall Viscosity uisng method proposed by Ian Bell 
    """
    components = model.groups.components
    models = []
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674] # global parameters
    ξ_pure=zeros(length(z))

    for j = 1:length(z)
        model_pure=SAFTgammaMie([components[j]])

        ξ_i = ["CH3" 0.484458;
        "CH2"  0.047793] # temporary manually tuned
        ξ = 0
        # GCM determination of ξ, doesn't yet include second order contributions
        groups = model_pure.groups.groups[1] #n-elemnet Vector{string}
        num_groups = model_pure.groups.n_groups[1] #n-element Vector{Int}
        for i in 1:length(groups)
            value = ξ_i[ξ_i[:, 1] .== groups[i], 2][1]
            ξ = ξ + value * num_groups[i]
            
            ξ_pure[j] = ξ
        end

    end
    ξ_mix=sum(z.*ξ_pure)
    R = Rgas()
    s_res=entropy_res(model,P,T,z)
    s_red=-s_res./R
    
    n_reduced= exp(n_g[1].*(s_red./ξ_mix) + n_g[2].*(s_red./ξ_mix).^(1.5) + n_g[3].*(s_red./ξ_mix).^(2) + n_g[4].*(s_red./ξ_mix).^(2.5)) -1

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar=molar_density(model,P,T,z)
    ρ_N = ρ_molar.*N_A

    Mw = Clapeyron.molecular_weight(model,z)
    m=Mw/N_A

    n_res = (n_reduced.*(ρ_N.^(2/3)).*sqrt.(m.*k_B.*T))./((s_red).^(2/3))

    viscosity = IB_CE_mix(model,T,z) + n_res
    return viscosity
end



model = SAFTgammaMie(["hexane","heptane"])

z = [1-0.6674, 0.6674]
z_hex=z[1]

N=100
P_range=LinRange(1e5,715e5,N)
T=303
P_bar=P_range./1e5


viscosity_mix=zeros(N)
for i in 1:N
    viscosity_mix[i] = IB_viscosity_mix(model,P_range[i],T,z)
end


# Plot viscosity vs. Pressure
plot(
    P_bar, viscosity_mix,
    xlabel = L"P\ \mathrm{/Bar}",
    ylabel = L"\eta\ \mathrm{/(Pa\cdot s)}",
    title  = "T=$(T) K, Hexane-Heptane Mixture, z_hex= $(z_hex)",
    lw = 2,
    label = "Bell Method",   # label for legend
    grid = false,
    guidefont = font(16)
)

##Experimental
#=
df = CSV.read("Hex_Hept Mix 303K x=0.4.csv", DataFrame)
P_experimental = df[:, 2]
n_experimental = df[:, 4] 
plot!(P_experimental./1e5, n_experimental;
      seriestype = :scatter,  
      label = false 
)  
=#
df = CSV.read("Hex_Hept Mix 303K x=0.4.csv", DataFrame)
P_experimental = df[:, 2]
n_experimental = df[:, 4] 
plot!(P_experimental./1e5, n_experimental,
      seriestype = :scatter,
      label = false
)  
#z = 0.4/100.21/(0.4/86.17848+0.6/100.21)
#z = 0.7/100.21/(0.3/86.17848+0.7/100.21)
# Calculate error
n_calc=zeros(length(P_experimental))
for i in 1:length(P_experimental)
    n_calc[i] = IB_viscosity_mix(model,P_experimental[i],T,z)
end

ADD = sum(abs.( (n_experimental .- n_calc)./n_experimental ))/length(P_experimental)
println("ADD = ", ADD)

