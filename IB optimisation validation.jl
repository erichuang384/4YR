using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays
include("bell_functions.jl")
#include("lotgering_functions.jl")



models = [
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie([("Nonane",["CH3"=>2,"CH2"=>7])]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie([("Undecane",["CH3"=>2,"CH2"=>9])]),
    SAFTgammaMie([("Dodecane",["CH3"=>2,"CH2"=>10])]),
    SAFTgammaMie([("Tridecane",["CH3"=>2,"CH2"=>11])]),
    SAFTgammaMie([("Tetradecane",["CH3"=>2,"CH2"=>12])]),
    SAFTgammaMie([("Pentadecane",["CH3"=>2,"CH2"=>13])]),
    SAFTgammaMie([("Hexadecane",["CH3"=>2,"CH2"=>14])]),
    SAFTgammaMie([("Heptadecane",["CH3"=>2,"CH2"=>15])])
]

#model_pentadecane = SAFTgammaMie(["pentadecane"])

#models = [model_pentane, model_hexane, model_heptane, model_octane, model_nonane, model_decane]

labels = ["Octane","Nonane", "Decane" ,"Undecane","Dodecane", "Tridecane","Tetradecane","Pentadecane", "Hexadecane","Heptadecane"]

#experimental values
exp_nonane = CSV.read("Validation Data/Nonane DETHERM.csv", DataFrame)
exp_undecane = CSV.read("Validation Data/Undecane DETHERM.csv", DataFrame)
exp_tridecane = CSV.read("Validation Data/Tridecane DETHERM.csv", DataFrame)
exp_pentadecane = CSV.read("Validation Data/Pentadecane DETHERM.csv", DataFrame)
exp_heptadecane = CSV.read("Validation Data/Heptadecane DETHERM.csv", DataFrame)

exp_octane = CSV.read("Training Data/Octane DETHERM.csv", DataFrame)
exp_decane =  CSV.read("Training Data/Decane DETHERM.csv", DataFrame)
exp_dodecane =  CSV.read("Training Data/Dodecane DETHERM.csv", DataFrame)
exp_tetradecane =  CSV.read("Training Data/Tetradecane DETHERM.csv", DataFrame)
exp_hexadecane =  CSV.read("Training Data/Hexadecane DETHERM.csv", DataFrame)


exp_data = [exp_octane, exp_nonane, exp_decane, exp_undecane, exp_dodecane, exp_tridecane, exp_tetradecane, exp_pentadecane, exp_hexadecane, exp_heptadecane] 

AAD = zeros(length(models))

#AAD_pentadecane = zeros(length(exp_pentadecane[:,1]))
for i in 1:length(models)
    T_exp = exp_data[i][:,2]
    n_exp = exp_data[i][:,3]
    P_exp = exp_data[i][:,1] 
    n_calc = IB_viscosity.(models[i],P_exp,T_exp) 

    AAD[i] = sum(abs.( (n_exp .- n_calc)./n_exp))/length(P_exp)
end
println("AAD = ", AAD)

#AAD_pentadecane = abs.(exp_pentadecane[:,3] .- IB_viscosity.(models[8],exp_pentadecane[:,1],exp_pentadecane[:,2]))./exp_pentadecane[:,3]

#exp_pentadecane.AAD = AAD_pentadecane
#CSV.write("Validation Data/Pentadecane DETHERM.csv", exp_pentadecane)

#AAD_heptadecane = abs.(exp_heptadecane[:,3] .- IB_viscosity.(models[10],exp_heptadecane[:,1],exp_heptadecane[:,2]))./exp_heptadecane[:,3]

#exp_heptadecane.AAD = AAD_heptadecane
#CSV.write("Validation Data/Heptadecane DETHERM.csv", exp_heptadecane)

#AAD_hexadecane = abs.(exp_hexadecane[:,3] .- IB_viscosity.(models[9],exp_hexadecane[:,1],exp_hexadecane[:,2]))./exp_hexadecane[:,3]

#exp_hexadecane.AAD = AAD_hexadecane
#CSV.write("Training Data/Hexadecane DETHERM.csv", exp_hexadecane)

AAD_octane = abs.(exp_hexadecane[:,3] .- IB_viscosity.(models[9],exp_hexadecane[:,1],exp_hexadecane[:,2]))./exp_hexadecane[:,3]

exp_hexadecane.AAD = AAD_hexadecane
CSV.write("Training Data/Hexadecane DETHERM.csv", exp_hexadecane)


#Scatter of AAD vs entropy for pentadecane

T_exp = exp_data[8][:,2]
n_exp = exp_data[8][:,3]
P_exp = exp_data[8][:,1] 
n_calc = IB_viscosity.(models[8],P_exp,T_exp) 

AAD = abs.( (n_exp .- n_calc)./n_exp).*100
res_ent = entropy_res.(models[8],P_exp,T_exp)./(-Rgas())

AAD_res_ent = scatter(res_ent,AAD,
    xlabel= L"s^+",
    ylabel = L"AAD\%",
    label = false,
    grid = false)




p_all = []  # store plots

for i in 1:length(models)
    model = models[i]
    data = exp_data[i]

    P_exp = data[:,1]
    T_exp = data[:,2]
    n_exp = data[:,3]

    n_calc = IB_viscosity.(model, P_exp, T_exp)
    AAD = ((n_exp .- n_calc) ./ n_exp) .* 100

    res_ent = entropy_res.(model, P_exp, T_exp) ./ (-Rgas())

    p = scatter(
        res_ent, AAD,
        xlabel = L"s^+",
        ylabel = L"AD (\%)",
        title = "AD% vs Residual Entropy for $(labels[i])",
        legend = false,
        markersize = 5,
        color = :blue
    )

    push!(p_all, p)
end
p_all[1]
p_all[2]
p_all[3]
p_all[4]
p_all[5]
p_all[6]
p_all[7]
p_all[8]
p_all[9]
p_all[10]

#savefig(p_all[1],"AAD vs res ent Octane")
savefig(p_all[5],"AD vs res ent Dodecane")

