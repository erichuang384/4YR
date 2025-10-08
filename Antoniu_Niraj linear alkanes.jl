using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings

include("all_functions.jl")

#model_butane = SAFTgammaMie(["butane"])
model_pentane = SAFTgammaMie(["pentane"])
model_hexane = SAFTgammaMie(["hexane"])
model_heptane = SAFTgammaMie(["heptane"])
model_octane = SAFTgammaMie(["octane"])
model_nonane = SAFTgammaMie(["nonane"])
model_decane = SAFTgammaMie(["decane"])
#model_pentadecane = SAFTgammaMie(["pentadecane"])

models = [model_pentane, model_hexane, model_heptane, model_octane, model_nonane, model_decane]

labels = ["pentane", "hexane","heptane","octane","nonane","decane"]

N=500
T_range=LinRange(250,400,N)
P=1e5

T_boil = saturation_temperature.(models[:],P)

#property calculations
viscosity=zeros(N,length(models))
for i in 1:length(models)
    vicosity_reduced= reduced_viscosity.(models[i],P,T_range[:])
    viscosity_CE = dilute_gas_viscosity.(models[i],T_range[:]) 
    viscosity[:,i] = vicosity_reduced.*viscosity_CE
end

#experimental values
exp_pentane = CSV.read("Experimental Data/pentane_1bar.csv", DataFrame)
exp_hexane = CSV.read("Experimental Data/hexane_1bar.csv", DataFrame)
exp_heptane = CSV.read("Experimental Data/heptane_1bar.csv", DataFrame)
exp_octane = CSV.read("Experimental Data/octane_1bar.csv", DataFrame)
exp_nonane = CSV.read("Experimental Data/nonane_1bar.csv", DataFrame)
exp_decane = CSV.read("Experimental Data/decane_1bar.csv", DataFrame)

exp_data = [exp_pentane, exp_hexane, exp_heptane, exp_octane, exp_nonane, exp_decane]

# Create a plot for each alkane
plots = []

for i in 1:length(models)
    g = plot(
        T_range, viscosity[:, i],
        lw = 2.5,
        label = labels[i],
        xlabel = L"T\;(\mathrm{K})",
        ylabel = L"\eta\;(\mathrm{Pa\cdot s})",
        title = "Viscosity of $(labels[i]) using SAFT-γ Mie",
        xlims = (250, 450)
    )
    push!(plots, g)
end

# Plot experimental

for i in 1:length(exp_data)
    scatter!(plots[i],exp_data[i][:,1],exp_data[i][:,3], label = false, lw=:2)
end
plots