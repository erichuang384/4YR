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

labels = ["Pentane", "Hexane","Heptane","Octane","Nonane","Decane"]

N=1000
T_range=LinRange(200,450,N)
P=1e5
T_boil = saturation_temperature.(models[:],P)

"""
T_boil = zeros(length(models))
for i in 1:length(models)
    dum_t = saturation_temperature(models[i],P)
    T_boil[i] = dum_t[1]
end

T_range = LinRange.(200,T_boil[:],N)
"""

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
        label = false,
        xlabel = L"T/\mathrm{K}",
        ylabel = L"\eta/\;(\mathrm{Pa\cdot s})",
        title = "$(labels[i])",
        xlims = (200,T_boil[i][1]),
        xguidefont = font(16),
        yguidefont = font(16)
    )
    push!(plots, g)
end

# Plot experimental

for i in 1:length(exp_data)
    scatter!(plots[i],exp_data[i][:,1],exp_data[i][:,3], label = false, lw=:2)
end
plots[1]

ylims!(plots[1],0,0.001)
ylims!(plots[2],0,0.0015)
ylims!(plots[3],0,0.0025)
ylims!(plots[4],0,0.0015)
ylims!(plots[5],0,0.002)
ylims!(plots[6],0,0.001)

savefig(plots[1],"pentane_1bar")
savefig(plots[2],"hexane_1bar")
savefig(plots[3],"heptane_1bar")
savefig(plots[4],"octane_1bar")
savefig(plots[5],"nonane_1bar")
savefig(plots[6],"decane_1bar")