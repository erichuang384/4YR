using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, StaticArrays
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))

models = [SAFTgammaMie(["Cyclopentane"]),
        SAFTgammaMie(["Cyclohexane"])]

exp_cyclopentane = CSV.read("Validation Data/Cyclopentane 1 bar.csv",DataFrame)
exp_cyclohexane = CSV.read("Validation Data/Cyclohexane 1 bar.csv",DataFrame)

exp_data = [exp_cyclopentane,
exp_cyclohexane]

p = 101000
N = 500
T_range = [LinRange(minimum(exp_data[1][:,2]),maximum(exp_data[1][:,2]),N), LinRange(minimum(exp_data[2][:,2]),maximum(exp_data[2][:,2]),N)]
#visc_calc = IB_viscosity_3param.(model,p,T_range)


visc_calc_cpent = IB_viscosity.(models[1],p,T_range[1])
visc_calc_cpent_T = IB_viscosity_3param_T.(models[1],p,T_range[1])

visc_calc_chex = IB_viscosity.(models[2],p,T_range[2])
visc_calc_chex_T = IB_viscosity_3param_T.(models[2],p,T_range[2])


plot_cpent_1bar = scatter(exp_cyclopentane[:,2], exp_cyclopentane[:,3],
	grid = false,
	label = false,
    color =:blue,
    marker =:diamond,
	xlabel = L"\textrm{T/K}",
	ylabel = L"\eta/\textrm{Pa s}",
    title = "Cyclopentane 1 bar")
    plot!(plot_cpent_1bar,T_range[1],visc_calc_cpent,
    color =:blue,
    label = false,
    lw=:3)
    plot!(plot_cpent_1bar,T_range[1],visc_calc_cpent_T,
    color =:green,
    label = "Temperature Dependent",
    lw=:3)

plot_chex_1bar = scatter(exp_cyclohexane[:,2], exp_cyclohexane[:,3],
	grid = false,
	label = false,
    color =:blue,
    marker =:diamond,
	xlabel = L"\textrm{T/K}",
	ylabel = L"\eta/\textrm{Pa s}",
    title = "Cyclohexane 1 bar")
    plot!(plot_chex_1bar,T_range[2],visc_calc_chex,
    color =:blue,
    label = false,
    lw=:3)
    
    plot!(plot_chex_1bar,T_range[2],visc_calc_chex_T,
    color =:green,
    label = "Temperature Dependent",
    lw=:3)

# AD vs s+


exp_cyclopentane = CSV.read("Training DATA/Cyclopentane DETHERM.csv",DataFrame)
exp_cyclohexane = CSV.read("Training DATA/Cyclohexane DETHERM.csv",DataFrame)

exp_data = [exp_cyclopentane,
exp_cyclohexane]

p_all = []  # store plots
labels = ["Cyclopentane" "Cyclohexane"]

for i in 1:length(models)
    model = models[i]
    data = exp_data[i]

    P_exp = data[:,1]
    T_exp = data[:,2]
    n_exp = data[:,3]

    n_calc = IB_viscosity_test.(model, P_exp, T_exp,0.76134)
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

# AAD calculations
# of training data
train_data_chex = CSV.read("Training DATA/Cyclohexane DETHERM.csv",DataFrame)
visc_train_chex = IB_viscosity_test.(models[2],train_data_chex[:,1],train_data_chex[:,2])
AAD_train_chex = sum(abs.(train_data_chex[:,3] .- visc_train_chex)./train_data_chex[:,3])/length(visc_train_chex)
#0.267 for T, 0.104 for no T

train_data_cpent = CSV.read("Training DATA/Cyclopentane DETHERM.csv",DataFrame)
visc_train_cpent = IB_viscosity_test.(models[1],train_data_cpent[:,1],train_data_cpent[:,2])
AAD_train_cpent = sum(abs.(train_data_cpent[:,3] .- visc_train_cpent)./train_data_cpent[:,3])/length(visc_train_cpent)
#0.126 for T, 1.65 for no T


#AAD of 1 bar untrained data
visc_untrain = IB_viscosity_3param_T.(model,exp_data[:,1],exp_data[:,2])
AAD_1bar = sum(abs.(exp_data[:,3] .- visc_untrain)./exp_data[:,3])/length(visc_untrain)
plot_benzene_1bar

savefig(plot_benzene_1bar,"Benzene 1 bar Temp xi")
print(AAD_train)
print(AAD_1bar)
plot_benzene_1bar