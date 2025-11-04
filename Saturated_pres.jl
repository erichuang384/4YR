# For data which doesn't have pressure given
using Statistics, DataFrames, Plots, LaTeXStrings, CSV, StaticArrays, Clapeyron
models = [
    SAFTgammaMie(["2-methylnonane"]),
    SAFTgammaMie(["4-methylnonane"])
]

exp_data = [
    CSV.read("Training DATA/Branched Alkane/2-methylnonane.csv", DataFrame),
    CSV.read("Training DATA/Branched Alkane/4-methylnonane.csv", DataFrame)]



sat_p_2 = saturation_pressure.(models[1],exp_data[1][:,1])
sat_p_4 = saturation_pressure.(models[2],exp_data[2][:,1])

#compare density
rho_2 = zeros(length(sat_p_2))
rho_4 = zeros(length(sat_p_4))
for i in 1:length(sat_p_2)
    rho_2[i] = IB_viscosity_test(models[1],sat_p_2[i][1],exp_data[1][i,1])
    rho_4[i] = IB_viscosity_test(models[2],sat_p_2[i][2],exp_data[2][i,1])
end

AAD_2 = sum(abs.(exp_data[1][:,2] .- rho_2) ./ exp_data[1][:,2]) / length(rho_2)
AAD_4 = sum(abs.(exp_data[2][:,2] .- rho_4) ./ exp_data[2][:,2]) / length(rho_4)

# PLOTTING
AAD_2_plot = (abs.(exp_data[1][:,2] .- rho_2) ./ exp_data[1][:,2])
AAD_4_plot = (abs.(exp_data[2][:,2] .- rho_4) ./ exp_data[2][:,2])

res_ent_2 = zeros(length(rho_2))
res_ent_4 = zeros(length(rho_4))
for i in 1:length(rho_2)
    res_ent_2[i] = entropy_res.(models[1],sat_p_2[i][1],exp_data[1][i,1]) ./ (-Rgas())
    res_ent_4[i] = entropy_res.(models[2],sat_p_4[i][1],exp_data[2][i,1]) ./ (-Rgas())
end

aad_plot_2 = scatter(
        res_ent_2, AAD_2_plot,
        xlabel = L"s^+",
        ylabel = L"AD (\%)",
        title = "AD% vs Residual Entropy for 2-methylnonane",
        legend = false,
        markersize = 5,
        color = :blue
    )

aad_plot_4 = scatter(
        res_ent_4, AAD_4_plot,
        xlabel = L"s^+",
        ylabel = L"AD (\%)",
        title = "AD% vs Residual Entropy for 4-methylnonane",
        legend = false,
        markersize = 5,
        color = :blue
    )