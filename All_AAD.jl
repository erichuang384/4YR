using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays
#include("model development.jl")

models = [
    SAFTgammaMie(["Butane"]),
	SAFTgammaMie(["Pentane"]),
	SAFTgammaMie(["Hexane"]),
	SAFTgammaMie(["Heptane"]),
	SAFTgammaMie(["Octane"]),
	SAFTgammaMie(["Nonane"]),
	SAFTgammaMie(["Decane"]),
	SAFTgammaMie(["Undecane"]),
	SAFTgammaMie(["Dodecane"]),
	SAFTgammaMie(["Tridecane"]),
	SAFTgammaMie(["Tetradecane"]),
	SAFTgammaMie(["Pentadecane"]),
	SAFTgammaMie(["Hexadecane"]),
	SAFTgammaMie(["Heptadecane"]),
	SAFTgammaMie(["n-eicosane"]),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
#    SAFTgammaMie(["9-octylheptadecane"]), # very questionable data
    SAFTgammaMie(["squalane"]),
    SAFTgammaMie(["3,5-dimethylheptane"]),
    SAFTgammaMie(["2-methyloctane"]),
    SAFTgammaMie(["4-methyloctane"]),
    SAFTgammaMie(["3,6-dimethyloctane"]),
    SAFTgammaMie(["2-methylnonane"]),
    SAFTgammaMie(["2-methyldecane"]),
    SAFTgammaMie(["3-methylundecane"]),
    SAFTgammaMie(["2-methylpentadecane"]),
    SAFTgammaMie(["7-methylhexadecane"]),
    SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["heptamethylnonane"]),
    SAFTgammaMie(["2,2,4-trimethylhexane"])

]


data_paths = [
    "Training DATA/Butane DETHERM.csv",
    "Training DATA/Pentane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training Data/Heptane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Validation Data/Nonane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Validation Data/Undecane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Validation Data/Tridecane DETHERM.csv",
    "Training Data/Tetradecane DETHERM.csv",
    "Validation Data/Pentadecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv",
    "Validation Data/Heptadecane DETHERM.csv",
    "Training DATA/n-eicosane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
   "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
#    "Training DATA/Branched Alkane/9-octylheptadecane.csv",
    "Training DATA/Branched Alkane/squalane.csv",
    "Training DATA/Branched Alkane/3,5-dimethylheptane.csv",
    "Training DATA/Branched Alkane/2-methyloctane.csv",
    "Training DATA/Branched Alkane/4-methyloctane.csv",
    "Training DATA/Branched Alkane/3,6-dimethyloctane.csv",
    "Training DATA/Branched Alkane/2-methylnonane.csv",
    "Training DATA/Branched Alkane/2-methyldecane.csv",
    "Training DATA/Branched Alkane/3-methylundecane.csv",
    "Training DATA/Branched Alkane/2-methylpentadecane.csv",
    "Training DATA/Branched Alkane/7-methylhexadecane.csv",
    "Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,2,4,4,6,8,8-heptamethylnonane.csv",
    "Training DATA/Branched Alkane/2,2,4-trimethylhexane.csv"
    
]

substances = [model.groups.components[1] for model in models]

exp_data = [load_experimental_data(p) for p in data_paths]

AAD = zeros(length(models))

for i in 1:length(models)
    T_exp = exp_data[i][:,2]
    n_exp = exp_data[i][:,3]
    P_exp = exp_data[i][:,1] 
    n_calc = bell_lot_test.(models[i],P_exp,T_exp) 

    AAD[i] = sum(abs.( (n_exp .- n_calc)./n_exp))/length(P_exp)
end
println("AAD = ", AAD)
mean(AAD)
maximum(AAD)
mean(AAD[2:end-1])
maximum(AAD[2:end-1])

df = DataFrame(Substance = String[], Temperature_Range = String[], Pressure_Range = String[], Num_Data_Points = Int[], AAD = Float64[]
, Weighting = Float64[]
)

    
weightings = [0.8, 1.0,1.0,0.0, 1.0,0.0, 1.0,1.0,1.0,1.0, 0.0, 1.0, 1.0, 1.0, 0.8,
0.15, 0.35, 0.45, 0.55, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 , 0.3 , 0.3, 0.3, 0.4, 0.1]
for i in 1:length(models)
    substance = substances[i]
    T_exp = exp_data[i][:,2]
    P_exp = exp_data[i][:,1]
    t_min = minimum(T_exp)
    t_max = maximum(T_exp)
    p_min = minimum(P_exp)
    p_max = maximum(P_exp)
    t_range = "$t_min - $t_max"
    p_range = "$p_min - $p_max"
    num_points = length(P_exp)
    aad = AAD[i]
   Weighting = weightings[i]
    push!(df, (substance, t_range, p_range, num_points, aad, Weighting))
end

CSV.write("ALL AAD_weighting.csv", df)