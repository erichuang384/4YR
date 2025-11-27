models = [SAFTgammaMie(["2-methyloctane"]), SAFTgammaMie(["4-methyloctane"])]

data_files = ["Training DATA/Branched Alkane/2-methyloctane.csv",
    "Training DATA/Branched Alkane/4-methyloctane.csv"]

exp_data_2 = CSV.read(data_files[1],DataFrame)
exp_data_4 = CSV.read(data_files[2],DataFrame)

n = 500
T_range = LinRange(283.0,373.15,n)
viscosity_2 = bell_lot_test.(models[1],1e5,T_range)
viscosity_4 = bell_lot_test.(models[2],1e5,T_range)


plot1 = scatter(exp_data_2[:,2], exp_data_2[:,3],
    grid = false,
    label = "2-methyloctane",
    color =:red,
    marker =:diamond,
    xlabel = L"\textrm{T/K}",
    ylabel = L"\eta/\textrm{Pa s}",
    title = "P = 1 bar",
    xguidefontsize = 14,
    yguidefontsize = 14,)
    scatter!(plot1,exp_data_4[:,2], exp_data_4[:,3],
    label = "4-methyloctane",
    color =:blue)
    plot!(plot1,T_range, viscosity_2,
    label = false,
    color =:green,
    lw =:2.5)
