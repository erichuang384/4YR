using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays

model = SAFTgammaMie(["Hexane","Octane"])

x_hexane = [0.1042, 0.1995, 0.2959, 0.3973, 0.4949, 0.5975, 0.6917, 0.794, 0.8955]
z = hcat(x_hexane, 1 .- x_hexane)
exp_rho = 0.7027 #kg/L
Mw = [Clapeyron.molecular_weight(model, z[i, :]) for i in 1:size(z,1)]
exp_rho_mol = exp_rho .* 10^3 .* 10^3 ./ (Mw .* 1000) #in g/m^3  / g/mol -> mol/m^3
exp_vol_mol = 1 ./ exp_rho_mol

exp_pres = zeros(length(exp_rho_mol))
for i in eachindex(exp_pres)
    exp_pres[i] = Clapeyron.pressure(model,exp_vol_mol[i], 293.15, z[i,:])
end
exp_pres