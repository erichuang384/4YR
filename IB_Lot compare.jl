using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays

include("bell_functions.jl")

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

N=500
T_range=LinRange(145,450,N)
P=1e5
T_boil = saturation_temperature.(models[:],P)

#property calculations
viscosity = zeros(N,length(models))
viscosity_IB = zeros(N,length(models))

#Lotgering
for i in 1:length(models)
    viscosity[:,i] = Lotgering_viscosity.(models[i],P,T_range[:]) 
end

for i in 1:length(models)
    viscosity_IB[:,i] = IB_viscosity.(models[i],P,T_range[:]) 
end

#experimental values
exp_pentane = CSV.read("Experimental Data/pentane_1bar.csv", DataFrame)
exp_hexane = CSV.read("Experimental Data/hexane_1bar.csv", DataFrame)
exp_heptane = CSV.read("Experimental Data/heptane_1bar.csv", DataFrame)
exp_octane = CSV.read("Experimental Data/octane_1bar.csv", DataFrame)
exp_nonane = CSV.read("Experimental Data/nonane_1bar.csv", DataFrame)
exp_decane = CSV.read("Experimental Data/decane_1bar.csv", DataFrame)

#NIST values
nist_octane = CSV.read("Experimental Data/octane_1bar_nist.csv", DataFrame)

exp_data = [exp_pentane, exp_hexane, exp_heptane, exp_octane, exp_nonane, exp_decane]

# Create a plot for each alkane
plots = []

#plot lotgering
for i in 1:length(models)
    g = plot(
        T_range, viscosity[:, i],
        lw = 5,
        label = "Lotgering",
        xlabel = L"T/\mathrm{K}",
        ylabel = L"\eta/\;(\mathrm{Pa\cdot s})",
        title = "$(labels[i])",
        xlims = (145,T_boil[i][1]),
        xguidefont = font(16),
        yguidefont = font(16),
        grid = false
    )
    push!(plots, g)
end


#plot IB
for i in 1:length(models)
    plot!(plots[i],T_range,viscosity_IB[:,i],label = "Bell", lw=:5)
end

# Plot experimental
for i in 1:length(exp_data)
    scatter!(plots[i],exp_data[i][:,1],exp_data[i][:,3], label = false, lw=:2)
end
plots[1]

ylims!(plots[1],0,0.004)

xlims!(plots[2],180,350)
ylims!(plots[2],0,0.0015)

xlims!(plots[3],180,350)
ylims!(plots[3],0,0.0025)


scatter!(plots[4],nist_octane[:,1],nist_octane[:,3],label = "NIST")
xlims!(plots[4],200,400)
ylims!(plots[4],0.,0.0015)

xlims!(plots[5],220,400)
ylims!(plots[5],0,0.002)

xlims!(plots[6],200,400)
ylims!(plots[6],0,0.003)

"""
savefig(plots[1],"pentane_1bar")
savefig(plots[2],"hexane_1bar")
savefig(plots[3],"heptane_1bar")
savefig(plots[4],"octane_1bar")
savefig(plots[5],"nonane_1bar")
savefig(plots[6],"decane_1bar")
"""