using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays
#include("model development.jl")

models = [
    #SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    #SAFTgammaMie(["2-methylnonane"]), NOT TP
    #SAFTgammaMie(["4-methylnonane"]), NOT TP
    #SAFTgammaMie(["heptamethylnonane"]),
    SAFTgammaMie(["squalane"])
]

data_paths = [
    #"Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
    #"Training DATA/Branched Alkane/2-methylnonane.csv",
    #"Training DATA/Branched Alkane/4-methylnonane.csv",
    #"Training DATA/Branched Alkane/heptamethylnonane.csv",
    "Training DATA/Branched Alkane/squalane.csv"
]

exp_data = [load_experimental_data(p) for p in data_paths]

AAD = zeros(length(models))

for i in 1:length(models)
    T_exp = exp_data[i][:,2]
    n_exp = exp_data[i][:,3]
    P_exp = exp_data[i][:,1] 
    n_calc = bell_lot_viscosity.(models[i],P_exp,T_exp) 

    AAD[i] = sum(abs.( (n_exp .- n_calc)./n_exp))/length(P_exp)
end
println("AAD = ", AAD)

#labels = ["2,6,10,14-tetramethylpentadecane", "Isobutane", "Isopentane", "Isohexane", "Squalane"]
labels = ["2,2,4-trimethylpentane", "2,6,10,14-tetramethylpentadecane", "Isobutane", "Isopentane", "Isohexane", "Heptamethylnonane", "Squalane"]
p_all = []  # store plots

for i in 1:length(models)
    model = models[i]
    data = exp_data[i]

    P_exp_plot = data[:,1]
    T_exp_plot = data[:,2]
    n_exp_plot = data[:,3]

    n_calc = bell_lot_viscosity.(model, P_exp_plot, T_exp_plot)
    AAD_plot = ((n_exp_plot .- n_calc) ./ n_exp_plot) .* 100

    res_ent = entropy_res.(model, P_exp_plot, T_exp_plot) ./ (-Rgas())

    p = scatter(
        res_ent, AAD_plot,
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



#savefig(p_all[1],"AAD vs res ent Octane")
#savefig(p_all[5],"AD vs res ent Dodecane")

