using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings

include("all_functions.jl")

model_butane = SAFTgammaMie(["butane"])
model_pentane = SAFTgammaMie(["pentane"])
model_hexane = SAFTgammaMie(["hexane"])
model_heptane = SAFTgammaMie(["heptane"])
model_octane = SAFTgammaMie(["octane"])
model_nonane = SAFTgammaMie(["nonane"])
model_decane = SAFTgammaMie(["decane"])
model_pentadecane = SAFTgammaMie(["pentadecane"])

models = [model_butane, model_pentane, model_hexane, model_heptane, model_octane, model_nonane, model_decane]

N=500
T_range=LinRange(250,450,N)
P=1e5
#property calculations
viscosity=zeros(N,length(models))
for i in 1:length(models)
    vicosity_reduced= reduced_viscosity.(models[i],P,T_range[:])
    viscosity_CE = dilute_gas_viscosity.(models[i],T_range[:]) 
    viscosity[:,i] = vicosity_reduced.*viscosity_CE
end

#experimental values
nist_hexane = CSV.read("Experimental Data/nist_hexane_1bar.csv", DataFrame)


#reduced_viscosity.(model_hexane,P,T_range[:])
labels = ["Butane", "Pentane", "Hexane", "Heptane","Octane", "Nonane", "Decane"]
graph = plot(T_range, viscosity[:,1], label=labels[1], lw=2)
for i in 2:length(models)
    plot!(graph,T_range, viscosity[:,i], label=labels[i], lw=2)
end
graph
xlabel!(L"T/ K")
ylabel!(L"\eta/ (Pa \cdot s)")
title!("Viscosity of Alkanes using SAFT-γ Mie")