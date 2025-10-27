using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings

include("bell_functions.jl")


#models = [model_pentane, model_hexane, model_heptane, model_octane, model_nonane, model_decane]

models = [
    SAFTgammaMie(["pentane"]),
    SAFTgammaMie(["hexane"]),
    SAFTgammaMie(["heptane"]),
    SAFTgammaMie(["octane"]),
    SAFTgammaMie(["nonane"]),
    SAFTgammaMie(["decane"]),
    SAFTgammaMie([("Undecane",["CH3"=>2,"CH2"=>9])]),
    SAFTgammaMie([("Dodecane",["CH3"=>2,"CH2"=>10])]),
    SAFTgammaMie([("Tridecane",["CH3"=>2,"CH2"=>11])]),
    SAFTgammaMie([("Tetradecane",["CH3"=>2,"CH2"=>12])]),
    SAFTgammaMie([("Pentadecane",["CH3"=>2,"CH2"=>13])]),
    SAFTgammaMie([("Hexadecane",["CH3"=>2,"CH2"=>14])])
]

labels = ["pentane","hexane","heptane", "octane", "Nonane", "Decane" ,"Undecane","Dodecane", "Tridecane","Tetradecane","Pentadecane", "Hexadecane"]


N=500
T_range=LinRange(145,450,N)
P=1e5
T_boil = saturation_temperature.(models[:],P)

#property calculations
viscosity = zeros(N,length(models))
viscosity_IB = zeros(N,length(models))

#Lotgering
for i in 1:length(models)
    viscosity[:,i] = IB_viscosity_test.(models[i],P,T_range[:]) 
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
exp_undecane = CSV.read("Experimental Data/undecane_1bar.csv", DataFrame)
exp_dodecane = CSV.read("Experimental Data/dodecane_1bar.csv", DataFrame)
exp_tridecane = CSV.read("Experimental Data/tridecane_1bar.csv", DataFrame)
exp_tetradecane = CSV.read("Experimental Data/tetradecane_1bar.csv", DataFrame)
exp_pentadecane = CSV.read("Experimental Data/pentadecane_1bar.csv", DataFrame)
exp_hexadecane = CSV.read("Experimental Data/hexadecane_1bar.csv", DataFrame)

#NIST values
nist_octane = CSV.read("Experimental Data/octane_1bar_nist.csv", DataFrame)

exp_data = [exp_pentane, exp_hexane, exp_heptane, exp_octane, exp_nonane, exp_decane, exp_undecane, exp_dodecane, exp_tridecane, exp_tetradecane, exp_pentadecane, exp_hexadecane]

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
    scatter!(plots[i],exp_data[i][:,2],exp_data[i][:,3], label = false, lw=:2)
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

xlims!(plots[7],250,400)
ylims!(plots[7],0,0.003)

xlims!(plots[8],250,450)
ylims!(plots[8],0,0.003)

xlims!(plots[9],250,450)
ylims!(plots[9],0,0.003)

xlims!(plots[10],250,450)
ylims!(plots[10],0,0.003)

xlims!(plots[11],250,450)
ylims!(plots[11],0,0.003)

xlims!(plots[12],250,450)
ylims!(plots[12],0,0.003)

#=
savefig(plots[1],"pentane_1bar")
savefig(plots[2],"hexane_1bar")
savefig(plots[3],"heptane_1bar")
savefig(plots[4],"octane_1bar")
savefig(plots[5],"nonane_1bar")
savefig(plots[6],"decane_1bar")
savefig(plots[7],"undecane_1bar")
savefig(plots[8],"dodecane_1bar")
savefig(plots[9],"tridecane_1bar")
savefig(plots[10],"tetradecane_1bar")
savefig(plots[11],"pentadecane_1bar")
savefig(plots[12],"hexadecane_1bar")
=#
plots[1]
plots[2]
plots[3]
plots[4]
plots[5]
plots[6]
plots[7]
plots[8]
plots[9]
plots[10]
plots[11]
plots[12]


AAD = zeros(length(models))
for i in 1:length(models)
    T_experimental = exp_data[i][:,2]
    n_experimental = exp_data[i][:,3]
    P_experimental = exp_data[i][:,1] 
    n_calculated = IB_viscosity_test.(models[i],P_experimental,T_experimental) 

    AAD[i] = sum(abs.( (n_experimental .- n_calculated)./n_experimental))/length(P_experimental)
end
println("AAD = ", AAD)