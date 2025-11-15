using CSV, DataFrames
using Clapeyron
using Plots
using Printf, LaTeXStrings, Statistics
using CMAEvolutionStrategy
using Random

# ------------------------------------------------------------
# 1. Model and data paths
# ------------------------------------------------------------
models = [
    SAFTgammaMie(["Pentane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Octane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Decane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Pentadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    SAFTgammaMie(["squalane"])
]

data_files = [
    "Training DATA/Pentane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Validation Data/Tridecane DETHERM.csv",
    "Validation Data/Pentadecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
    "Training DATA/Branched Alkane/squalane.csv"
]

# ------------------------------------------------------------
# 2. Reduced viscosity helper
# ------------------------------------------------------------
function reduced_visc(model::EoSModel, P, T, visc)
    visc_CE = IB_CE(model, T)
    s_res = entropy_res(model, P, T)
    R = Clapeyron.Rgas()
    s_red = -s_res / R
    total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])
    z_term = (-s_res ./ R) ./ total_sf

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A
    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A
    n_reduced = (visc .- visc_CE) ./ ((ρ_N .^ (2/3)) .* sqrt.(m .* k_B .* T)) .* (s_red .^ (2/3))
    return n_reduced, z_term
end

# ------------------------------------------------------------
# 3. Precompute dataset & group info
# ------------------------------------------------------------
model_names = [m.groups.components[1] for m in models]

data_z_list = Vector{Vector{Float64}}(undef, length(models))
data_y_list = Vector{Vector{Float64}}(undef, length(models))

num_CH3_list = zeros(length(models))
num_CH2_list = zeros(length(models))
num_CH_list  = zeros(length(models))
S_CH3_list = zeros(length(models))
S_CH2_list = zeros(length(models))
S_CH_list  = zeros(length(models))
sigma_CH3_list = zeros(length(models))
sigma_CH2_list = zeros(length(models))
sigma_CH_list  = zeros(length(models))
m_gc_list = zeros(length(models))
total_sf_list = zeros(length(models))
temp_list = Vector{Vector{Float64}}(undef, length(models))
epsilon_list = zeros(length(models))

total_points = Ref(0)

for (i, (model, file)) in enumerate(zip(models, data_files))
    try
        df = CSV.read(file, DataFrame)
        P, T, visc_exp = df[:, 1], df[:, 2], df[:, 3]

        results = reduced_visc.(models[i], P, T, visc_exp)
        n_red = [r[1] for r in results]
        z_term = [r[2] for r in results]
        mask = isfinite.(n_red) .& isfinite.(z_term) .& (n_red .> -1)

        data_z_list[i] = z_term[mask]
        data_y_list[i] = log.(n_red[mask] .+ 1.0)

        # Get CH3, CH2, CH info
        group_names = model.groups.groups[1]
        n_groups = model.groups.n_groups[1]
        S_values = model.params.shapefactor.values
        σ_values = diag(model.params.sigma.values) .* 1e10

        for (g, n) in zip(group_names, n_groups)
            if g == "CH3"
                num_CH3_list[i] = n
                S_CH3_list[i] = S_values[group_names .== "CH3"][1]
                sigma_CH3_list[i] = σ_values[group_names .== "CH3"][1]
            elseif g == "CH2"
                num_CH2_list[i] = n
                S_CH2_list[i] = S_values[group_names .== "CH2"][1]
                sigma_CH2_list[i] = σ_values[group_names .== "CH2"][1]
            elseif g == "CH"
                num_CH_list[i] = n
                S_CH_list[i] = S_values[group_names .== "CH"][1]
                sigma_CH_list[i] = σ_values[group_names .== "CH"][1]
            end
        end

        m_gc_list[i] = sum(n_groups)
        total_sf_list[i] = sum(S_values .* n_groups)
        temp_list[i] = T
        epsilon_list[i] = ϵ_OFE(models[i])
        total_points[] += length(data_z_list[i])
    catch err
        @warn "Skipping dataset" file error=err
        data_z_list[i] = Float64[]
        data_y_list[i] = Float64[]
    end
end

println("Loaded $(total_points[]) valid points across $(length(models)) datasets.")

# ------------------------------------------------------------
# 4. Objective function (optimize CH + gamma + D_i)
# ------------------------------------------------------------
function sse_group_contrib(params::AbstractVector{<:Real})
    if length(params) != 3
        return 1e20
    end

    # Fixed CH3 and CH2 coefficients
    A_CH3, B_CH3, C_CH3 = -0.011201519044440274, 0.05455747701575829, -0.010387746016789993
    A_CH2, B_CH2, C_CH2 = -0.0020733761429243907, 0.041278859543724934, -0.004369819922319668
    gamma = 0.29776486431023386
    D_i = 0.004612286325777618

    # Optimized
    A_CH, B_CH, C_CH = params

    total = 0.0
    for i in eachindex(models)
        z = data_z_list[i]
        y = data_y_list[i]
        if isempty(z)
            continue
        end

        num_CH3 = num_CH3_list[i]
        num_CH2 = num_CH2_list[i]
        num_CH  = num_CH_list[i]
        S_CH3 = S_CH3_list[i]
        S_CH2 = S_CH2_list[i]
        S_CH  = S_CH_list[i]
        σ_CH3 = sigma_CH3_list[i]
        σ_CH2 = sigma_CH2_list[i]
        σ_CH  = sigma_CH_list[i]
        m_gc = m_gc_list[i]
        tot_sf = total_sf_list[i]
        T = temp_list[i]

        V = num_CH3*S_CH3*σ_CH3^3 + num_CH2*S_CH2*σ_CH2^3 + num_CH*S_CH*σ_CH^3
        n_g1 = A_CH3*S_CH3*σ_CH3^3*num_CH3 + A_CH2*S_CH2*σ_CH2^3*num_CH2 + A_CH*S_CH*σ_CH^3*num_CH
        n_g2 = (B_CH3*S_CH3*σ_CH3^3*num_CH3 + B_CH2*S_CH2*σ_CH2^3*num_CH2 + B_CH*S_CH*σ_CH^3*num_CH) / V^gamma
        n_g3 = (C_CH3 .* num_CH3 .+ C_CH2 .*num_CH2 .+ C_CH .*num_CH) .* (1 .+ log.(T))

        D = D_i*m_gc
        y_pred = n_g1 .+ n_g2.*z .+ n_g3.*(z.^2) .+ D.*(z.^3)

        η = exp.(y) .- 1
        η_pred = exp.(y_pred) .- 1
        total += sum(((η .- η_pred) ./ η).^2) / length(y)
    end
    return isfinite(total) ? total : 1e20
end

# ------------------------------------------------------------
# 5. CMA-ES Optimization
# ------------------------------------------------------------
println("\nStarting CMA-ES optimization (CH + γ + D_i)...")

x0 = [0.04, 0.06, 0.015]
σ0 = 0.01
seed = 42
Random.seed!(seed)

result = minimize(
    sse_group_contrib,
    x0,
    σ0;
    seed = seed,
    verbosity = 2,
    maxiter = 8000,
    stagnation = 3000,
    ftol = 1e-10
)

params_opt = xbest(result)
final_sse = fbest(result)

A_CH, B_CH, C_CH, gamma_opt, D_i_opt = params_opt
println("\n✅ CMA-ES Optimization Complete")
@printf "A_CH = %.8f\n" A_CH
@printf "B_CH = %.8f\n" B_CH
@printf "C_CH = %.8f\n" C_CH
@printf "gamma = %.8f\n" gamma_opt
@printf "D_i = %.8f\n" D_i_opt
@printf "Final SSE = %.8e\n" final_sse
