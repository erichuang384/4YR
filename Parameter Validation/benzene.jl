using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, StaticArrays
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))

model = SAFTgammaMie([("Benzene",["aCH"=>6])])

exp_data = CSV.read("Validation Data/Benzene 1 bar.csv",DataFrame)

p = 1e5
N = 300
T_range = LinRange(minimum(exp_data[:,2]),maximum(exp_data[:,2]),N)
#visc_calc = IB_viscosity_3param.(model,p,T_range)
visc_calc = IB_viscosity.(model,p,T_range)
visc_calc_temp = IB_viscosity_3param_T.(model,p,T_range)
plot_benzene_1bar = scatter(exp_data[:,2], exp_data[:,3],
	grid = false,
	label = false,
    color =:blue,
    marker =:diamond,
	xlabel = L"\textrm{T/K}",
	ylabel = L"\eta/\textrm{Pa s}",
    title = "Benzene 1 bar")
    plot!(plot_benzene_1bar,T_range,visc_calc,
    color =:blue,
    label = false,
    lw=:3)
    plot!(plot_benzene_1bar,T_range,visc_calc_temp,
    color =:green,
    label = "Temperature xi",
    lw=:3)

# AAD 
exp_data = CSV.read("Training DATA/Benzene DETHERM.csv",DataFrame)

data = exp_data

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
        title = "AD% vs Residual Entropy for Benzene",
        legend = false,
        markersize = 5,
        color = :blue
    )










# of training data
train_data = CSV.read("Training DATA/Benzene DETHERM.csv",DataFrame)
visc_train = IB_viscosity_3param_T.(model,train_data[:,1],train_data[:,2])
AAD_train = sum(abs.(train_data[:,3] .- visc_train)./train_data[:,3])/length(visc_train)

#AAD of 1 bar untrained data
visc_untrain = IB_viscosity_3param_T.(model,exp_data[:,1],exp_data[:,2])
AAD_1bar = sum(abs.(exp_data[:,3] .- visc_untrain)./exp_data[:,3])/length(visc_untrain)
plot_benzene_1bar

#savefig(plot_benzene_1bar,"Benzene 1 bar Temp xi")
print(AAD_train)
print(AAD_1bar)
plot_benzene_1bar