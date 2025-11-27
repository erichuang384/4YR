[-1.7959349402713636, 4.359772186971961, -0.8308696445362812, 0.29008728624727087, 1.2400951096804953] , 0.5# a b c d m 
 [-4.031352527160982, 7.038736176803892, -0.0010626009022930566, 0.4256611440691021, 2690.8132467500022, 0.13175630909917577]# a b c d m p



function bell_lot_empirical(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
 
#    A, B, C, D, m , p = -4.031352527160982, 7.038736176803892, -0.0010626009022930566, 0.4256611440691021, 2690.8132467500022, 0.13175630909917577

    A, B, C, D, m, p = -1.7959349402713636, 4.359772186971961, -0.8308696445362812, 0.29008728624727087, 1.2400951096804953, 0.5

    n_g_1 = A
    n_g_2 = B

    n_g_4 = D

    crit_point = crit_pure(model)
    T_C = crit_point[1]

    n_g_3 = C * (1 .+ m .*(T ./ T_C) .^ p)

    S = model.params.shapefactor.values
    num_groups = model.groups.n_groups[1]
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
    #Z = (-s_res) / (R)
 
    ln_n_reduced = n_g_1 + n_g_2 * Z + n_g_3 * Z^2 + n_g_4 * Z^3
 
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
 
    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar .* N_A
 
    Mw = Clapeyron.molecular_weight(model, z)
 
    m = Mw / N_A
    n_reduced = exp(ln_n_reduced) - 1.0
 
    n_res = (n_reduced) .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)# ./ ((s_red) .^ (2 / 3))
 
    viscosity = n_res + IB_CE(model, T)
    return viscosity
end

using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays
#include("model development.jl")

models = [
    SAFTgammaMie(["squalane"])
]


data_paths = [
    "Training DATA/Branched Alkane/squalane.csv"
]

substances = [model.groups.components[1] for model in models]

exp_data = [load_experimental_data(p) for p in data_paths]

AAD = zeros(length(models))

AAD_list = Vector{Vector{Float64}}(undef, length(models))

for i in 1:length(models)
    T_exp = exp_data[i][:,2]
    n_exp = exp_data[i][:,3]
    P_exp = exp_data[i][:,1] 
    n_calc = bell_lot_empirical.(models[i],P_exp,T_exp)
    AAD_list[i] = abs.( (n_exp .- n_calc)./n_exp)
    AAD[i] = sum(abs.( (n_exp .- n_calc)./n_exp))/length(P_exp)
end
println("AAD = ", AAD)
maximum(AAD_list[1])
std(AAD_list[1])