using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings

include("all_functions.jl")

model_hexane=SAFTgammaMie(["hexane"])
#n_hexane = reduced_viscosity(model_hexane,1e5,320)   

N=100
T_range=LinRange(250,450,N)
P=20e5

#n_reduced=zeros(N)
#n_CE=zeros(N)
viscosity=zeros(N)
for i in 1:N
    vicosity_reduced= reduced_viscosity(model_hexane,P,T_range[i])
    viscosity_CE = dilute_gas_viscosity(model_hexane,T_range[i]) 
    viscosity[i]=vicosity_reduced.*viscosity_CE
end

# Plot viscosity vs. temperature
plot(
    T_range, viscosity,
    xlabel = L"T\ \mathrm{(K)}",
    ylabel = L"\eta\ \mathrm{(Pa\,s)}",
    title  = "Viscosity of Hexane at 20 bar",
    lw = 2,
    grid = false,
    legend = false,
)