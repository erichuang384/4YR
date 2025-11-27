# ==========================================
# Single m per model (CH3/CH2/CH/C included), full model set from your first script
# User-editable constants block (no const), CMA-ES optimizes only m_i
# ==========================================

using CSV, DataFrames
using Clapeyron
using Plots
using Printf, LaTeXStrings, Statistics
using CMAEvolutionStrategy
using Random

# -----------------------------------
# Models and data (use your full list)
# -----------------------------------


models = [
    # SAFTgammaMie(["Ethane"]),
    SAFTgammaMie(["Butane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Pentane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Octane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Decane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = BasicIdeal),
 #   SAFTgammaMie(["Pentadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Heptadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Eicosane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
    # SAFTgammaMie(["9-octylheptadecane"]), # very questionable data
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

data_files = [
    # "Training DATA/ethane.csv",
    "Training DATA/Butane DETHERM.csv",
    "Training DATA/Pentane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Validation Data/Tridecane DETHERM.csv",
 #   "Validation Data/Pentadecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv",
    "Validation Data/Heptadecane DETHERM.csv",
    "Training DATA/n-eicosane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
    # "Training DATA/Branched Alkane/9-octylheptadecane.csv",
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

# Dataset weights (from your first script; length must match models)
weightings = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
    0.15, 0.35, 0.45, 0.55, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.4, 0.1
]

# -----------------------------------
# Reduced viscosity calculation
# -----------------------------------
function reduced_visc(model::EoSModel, P, T, visc)
    visc_CE = IB_CE(model, T)
    s_res   = entropy_res(model, P, T)
    R       = Clapeyron.Rgas()
    total_sf = sum(model.params.shapefactor.values.* model.groups.n_groups[1])

    # residual-entropy abscissa
    z_term = (-s_res./ R)./ total_sf

    # reduced residual viscosity scaling
    N_A     = Clapeyron.N_A
    k_B     = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N     = ρ_molar.* N_A
    Mw      = Clapeyron.molecular_weight(model)
    m       = Mw / N_A
    n_reduced = (visc.- visc_CE)./ ((ρ_N.^ (2/3)).* sqrt.(m.* k_B.* T))
    return n_reduced, z_term
end

# -----------------------------------
# Precompute datasets and group data
# -----------------------------------
model_names       = [m.groups.components[1] for m in models]
data_z_list       = Vector{Vector{Float64}}(undef, length(models))
data_y_list       = Vector{Vector{Float64}}(undef, length(models))
data_T_list       = Vector{Vector{Float64}}(undef, length(models))

crit_pure_list    = Vector{Float64}(undef, length(models))
crit_pres_list =   Vector{Float64}(undef, length(models))
crit_entr_res_list = Vector{Float64}(undef, length(models))
acentric_list     = Vector{Float64}(undef, length(models))

# Group counts and parameters
ch3_count_list    = Vector{Float64}(undef, length(models))
ch2_count_list    = Vector{Float64}(undef, length(models))
ch_count_list     = Vector{Float64}(undef, length(models))
c_count_list      = Vector{Float64}(undef, length(models))

S_ch3_model_list  = Vector{Float64}(undef, length(models))
S_ch2_model_list  = Vector{Float64}(undef, length(models))
S_ch_model_list   = Vector{Float64}(undef, length(models))
S_c_model_list    = Vector{Float64}(undef, length(models))

sigma_ch3_list    = Vector{Float64}(undef, length(models))
sigma_ch2_list    = Vector{Float64}(undef, length(models))
sigma_ch_list     = Vector{Float64}(undef, length(models))
sigma_c_list      = Vector{Float64}(undef, length(models))

total_points = Ref(0)

for (i, (model, file)) in enumerate(zip(models, data_files))
    try
        exp_data = CSV.read(file, DataFrame)
        P, T, visc_exp = exp_data[:, 1], exp_data[:, 2], exp_data[:, 3]

        results  = reduced_visc.(models[i], P, T, visc_exp)
        n_red    = [r[1] for r in results]
        z_term   = [r[2] for r in results]

        mask = isfinite.(n_red).& isfinite.(z_term).& (n_red.> -1)

        data_z_list[i] = z_term[mask]
        data_y_list[i] = log.(n_red[mask].+ 1.0)
        data_T_list[i] = T[mask]

        # Critical temperature
        crit_point        = crit_pure(model)
        crit_pure_list[i] = crit_point[1]
        crit_pres_list[i] = crit_point[2]
        crit_entr_res_list[i] = -entropy_res(model, crit_point[2], crit_point[1])/Rgas()

        # Acentric factor
        acentric_list[i]  = acentric_factor(model)

        # Extract group data
        groups = model.groups.groups[1]           # group names
        ng     = model.groups.n_groups[1]         # counts per group
        Svals  = model.params.shapefactor.values  # shapefactors aligned with groups
        σdiag  = diag(model.params.sigma.values)  # sigma per group [m]
        # group indices
        i_ch3 = findfirst(==("CH3"), groups)
        i_ch2 = findfirst(==("CH2"), groups)
        i_ch  = findfirst(==("CH"),  groups)
        i_c   = findfirst(==("C"),   groups)

        # counts
        ch3_count_list[i] = i_ch3 === nothing ? 0.0 : ng[i_ch3]
        ch2_count_list[i] = i_ch2 === nothing ? 0.0 : ng[i_ch2]
        ch_count_list[i]  = i_ch  === nothing ? 0.0 : ng[i_ch]
        c_count_list[i]   = i_c   === nothing ? 0.0 : ng[i_c]

        # shapefactors
        S_ch3_model_list[i] = i_ch3 === nothing ? 0.0 : Svals[i_ch3]
        S_ch2_model_list[i] = i_ch2 === nothing ? 0.0 : Svals[i_ch2]
        S_ch_model_list[i]  = i_ch  === nothing ? 0.0 : Svals[i_ch]
        S_c_model_list[i]   = i_c   === nothing ? 0.0 : Svals[i_c]

        # sigmas in Å (then cube later)
        sigma_ch3_list[i] = i_ch3 === nothing ? 0.0 : σdiag[i_ch3] * 1e10
        sigma_ch2_list[i] = i_ch2 === nothing ? 0.0 : σdiag[i_ch2] * 1e10
        sigma_ch_list[i]  = i_ch  === nothing ? 0.0 : σdiag[i_ch]  * 1e10
        sigma_c_list[i]   = i_c   === nothing ? 0.0 : σdiag[i_c]   * 1e10

        total_points[] += length(data_z_list[i])
    catch err
        @warn "Skipping invalid dataset" file error=err
        data_z_list[i]       = Float64[]
        data_y_list[i]       = Float64[]
        data_T_list[i]       = Float64[]
        crit_pure_list[i]    = NaN
        acentric_list[i]     = NaN
        ch3_count_list[i]    = NaN
        ch2_count_list[i]    = NaN
        ch_count_list[i]     = NaN
        c_count_list[i]      = NaN
        S_ch3_model_list[i]  = NaN
        S_ch2_model_list[i]  = NaN
        S_ch_model_list[i]   = NaN
        S_c_model_list[i]    = NaN
        sigma_ch3_list[i]    = NaN
        sigma_ch2_list[i]    = NaN
        sigma_ch_list[i]     = NaN
        sigma_c_list[i]      = NaN
    end
end

println("Loaded $(total_points[]) valid data points across $(length(models)) datasets.")
if total_points[] == 0
    error("No valid data points loaded — check file paths and data formats.")
end

# -----------------------------------
# USER-EDITABLE: Fixed GC constants (CH3/CH2/CH/C)
# Put your final numbers here (kept fixed during m_i optimization)
# -----------------------------------
# Example: paste your tuned constants here

A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, A_CH, B_CH, C_CH, A_C, B_C, C_C, gamma_i, D_CH3, D_CH2, D_CH, D_C = 0.18380429041626706, 0.00926271817583487, -0.016736791143933925, -1.1628779418673048, 0.00971090571491477, -0.013748613245391289, -7.74673863485519, 0.016728450066019186, -0.029417127854903555, 5.201552330552332, -0.06115356102481718, 0.05215664709797355, 0.142522263832269, 5.213959989501366e-5, 0.00018572431352956148, 0.0022188349152456053, 0.003361644345102922
# From m = 0
A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, A_CH, B_CH, C_CH, A_C, B_C, C_C, gamma_i, D_CH3, D_CH2, D_CH, D_C = 0.34043551434546326, 0.017012242567565377, -0.006887844044150973, -1.3623420568844034, 0.052649076342669916, -0.027731222449147202, -12.185401526729436, 0.2657407762968443, -0.10022829684011945, 8.58857876264099, -0.19780912678696613, 0.13386433365372696, 0.40208606511671974, 6.0527726076920334e-5, 0.00020790399486063813, 0.002683733001198193, 0.0009125056211291571

# Helper for A term (n_g1)
A_vdw_fixed(model) = A_vdw_opt(model, A_CH3, A_CH2, A_CH, A_C)

# -----------------------------------
# Build index of active datasets (with data)
# -----------------------------------
active_idx = findall(i -> !isempty(data_z_list[i]) && !isempty(data_y_list[i]), 1:length(models))
println("Active datasets: $(length(active_idx)) / $(length(models))")

# -----------------------------------
# Objective: optimize only m_i (one value per model with data), others fixed
# -----------------------------------
function sse_m_only(m_params::AbstractVector{<:Real})
    if length(m_params) != length(active_idx)
        return 1e20
    end

    total = 0.0
    for (j, i) in enumerate(active_idx)
        z = data_z_list[i]
        y = data_y_list[i]
        T = data_T_list[i]
        if isempty(z); continue; end

        # Precomputed/group terms for model i
        ch3  = ch3_count_list[i]
        ch2  = ch2_count_list[i]
        ch   = ch_count_list[i]
        c    = c_count_list[i]

        S_ch3 = S_ch3_model_list[i]
        S_ch2 = S_ch2_model_list[i]
        S_ch  = S_ch_model_list[i]
        S_c   = S_c_model_list[i]

        sigma_ch3 = sigma_ch3_list[i]^3
        sigma_ch2 = sigma_ch2_list[i]^3
        sigma_ch  = sigma_ch_list[i]^3
        sigma_c   = sigma_c_list[i]^3

        Tc  = crit_pure_list[i]
        res_entr_crit = crit_entr_res_list[i] - 1.0


        # Volumes
        V_ch3 = ch3 * S_ch3 * sigma_ch3
        V_ch2 = ch2 * S_ch2 * sigma_ch2
        V_ch  = ch  * S_ch  * sigma_ch
        V_c   = c   * S_c   * sigma_c
        V_tot = V_ch3 + V_ch2 + V_ch + V_c
        if !(isfinite(V_tot)) || V_tot <= 0
            total += 1e20
            continue
        end

        # Fixed model terms
        n_g1 = A_vdw_fixed(models[i])
        n_g2 = (B_CH3 * V_ch3 + B_CH2 * V_ch2 + B_CH * V_ch + B_C * V_c) / V_tot^gamma_i

        # Single optimized scalar m_i for this model:
        m_i = m_params[j]

        # Temperature dependence: use cubic reduced temperature as in your later script
        θ = sqrt.((T)./ Tc)
        n_g3 = (C_CH3 * ch3 + C_CH2 * ch2 + C_CH * ch + C_C * c).* (1 .+ m_i.* θ)

        D    = (D_CH3 * V_ch3 + D_CH2 * V_ch2 + D_CH * V_ch + D_C * V_c)

        y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)

        if any(!isfinite, y_pred)
            total += 1e20
            continue
        end

        # normalized SSE on eta = exp(y) - 1 (dataset weighting applied)
        η      = exp.(y).- 1
        η_pred = exp.(y_pred).- 1
        w = weightings[i]
        total += (sum(((η.- η_pred)./ η).^ 2) / length(y)) * w
    end

    return isfinite(total) ? total : 1e20
end

println("\nStarting CMA-ES optimization of per-model m parameters (CH3/CH2/CH/C, all other params fixed)...")

# -----------------------------------
# CMA-ES setup: initial guess for m per active model
# -----------------------------------
m0_active = [0.379642 + 1.54226 * acentric_list[i] - 0.26992 * acentric_list[i]^2
             for i in active_idx]

σ0 = 0.01
seed = 142
Random.seed!(seed)
stagnation_iters = 5000
iter_counter = Ref(0)

result = minimize(
    sse_m_only,
    m0_active,
    σ0;
    seed = seed,
    verbosity = 2,
    stagnation = stagnation_iters,
    maxiter = 100000,
    ftol = 1e-12,
    callback = (opt, x, fx, ranks) -> begin
        iter_counter[] += 1
        if iter_counter[] % 50 == 0
            try
                println(@sprintf("Iter %d: fmin=%.6e best=%s",
                                 iter_counter[], minimum(fx), string(xbest(opt))))
            catch
                println(@sprintf("Iter %d: callback invoked", iter_counter[]))
            end
        end
    end
)

m_i_active = xbest(result)
final_sse  = fbest(result)

# Build full m_i vector aligned with 'models'
m_i_full = fill(NaN, length(models))
for (j, i) in enumerate(active_idx)
    m_i_full[i] = m_i_active[j]
end

println("\n✅ CMA-ES m_i optimization successful!")
@printf "Final SSE = %.8e\n", final_sse
println("Per-model m_i values:")
for (i, name) in enumerate(model_names)
    @printf("  %-30s  m_i = %10.6f\n", name, m_i_full[i])
end

# -----------------------------------
# Plot 1: All components — model vs data with optimized m_i
# -----------------------------------
plot1 = plot(
    grid = false,
    xlabel = L"z_{\mathrm{term}}",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Dimensionless Viscosity Collapse — CMA-ES (m_i-only, CH3/CH2/CH/C)",
    legend = :bottomright,
    lw = 2
)

# Scatter all experimental datapoints
for i in eachindex(models)
    z = data_z_list[i]
    y = data_y_list[i]
    if !isempty(z)
        scatter!(plot1, z, y, label=false, markersize=3, markerstrokewidth=0.0)
    end
end

# Overlay model prediction curves for each component (consistent with objective)
for i in eachindex(models)
    z = data_z_list[i]
    T = data_T_list[i]
    if isempty(z)
        continue
    end

    ch3  = ch3_count_list[i]
    ch2  = ch2_count_list[i]
    ch   = ch_count_list[i]
    c    = c_count_list[i]

    S_ch3 = S_ch3_model_list[i]
    S_ch2 = S_ch2_model_list[i]
    S_ch  = S_ch_model_list[i]
    S_c   = S_c_model_list[i]

    sigma_ch3 = sigma_ch3_list[i]^3
    sigma_ch2 = sigma_ch2_list[i]^3
    sigma_ch  = sigma_ch_list[i]^3
    sigma_c   = sigma_c_list[i]^3

    Tc  = crit_pure_list[i]

    V_ch3 = ch3 * S_ch3 * sigma_ch3
    V_ch2 = ch2 * S_ch2 * sigma_ch2
    V_ch  = ch  * S_ch  * sigma_ch
    V_c   = c   * S_c   * sigma_c
    V_tot = V_ch3 + V_ch2 + V_ch + V_c
    if !(isfinite(V_tot)) || V_tot <= 0
        continue
    end

    n_g1 = A_vdw_fixed(models[i])
    n_g2 = (B_CH3 * V_ch3 + B_CH2 * V_ch2 + B_CH * V_ch + B_C * V_c) / V_tot^gamma_i

    m_i = m_i_full[i]
    θ = ((T.- Tc)./ Tc).^ 3
    n_g3 = (C_CH3 * ch3 + C_CH2 * ch2 + C_CH * ch + C_C * c).* (1.+ m_i.* θ)

    D = (D_CH3 * V_ch3 + D_CH2 * V_ch2 + D_CH * V_ch + D_C * V_c)

    y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)

    idx = sortperm(z)
    plot!(plot1, z[idx], y_pred[idx], lw = 2, label = "$(model_names[i]) model")
end

display(plot1)

# -----------------------------------
# Plot 2: Single component detail — pick an index
# -----------------------------------
index = 16  # e.g., 16 for "squalane" in your list; change as desired
plot2 = plot(
    grid = false,
    xlabel = L"z_{\mathrm{term}}",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Single Component — CMA-ES (m_i-only, CH3/CH2/CH/C)",
    legend = :bottomright,
    lw = 2
)

z = data_z_list[index]
y = data_y_list[index]
T = data_T_list[index]
if !isempty(z)
    scatter!(plot2, z, y, label="data", markersize=4, markerstrokewidth=0.0)
end

ch3  = ch3_count_list[index]
ch2  = ch2_count_list[index]
ch   = ch_count_list[index]
c    = c_count_list[index]

S_ch3 = S_ch3_model_list[index]
S_ch2 = S_ch2_model_list[index]
S_ch  = S_ch_model_list[index]
S_c   = S_c_model_list[index]

sigma_ch3 = sigma_ch3_list[index]^3
sigma_ch2 = sigma_ch2_list[index]^3
sigma_ch  = sigma_ch_list[index]^3
sigma_c   = sigma_c_list[index]^3

Tc  = crit_pure_list[index]

V_ch3 = ch3 * S_ch3 * sigma_ch3
V_ch2 = ch2 * S_ch2 * sigma_ch2
V_ch  = ch  * S_ch  * sigma_ch
V_c   = c   * S_c   * sigma_c
V_tot = V_ch3 + V_ch2 + V_ch + V_c

n_g1 = A_vdw_fixed(models[index])
n_g2 = (B_CH3 * V_ch3 + B_CH2 * V_ch2 + B_CH * V_ch + B_C * V_c) / V_tot^gamma_i

m_i = m_i_full[index]
θ = ((T.- Tc)./ Tc).^ 3
n_g3 = (C_CH3 * ch3 + C_CH2 * ch2 + C_CH * ch + C_C * c).* (1.+ m_i.* θ)

D = (D_CH3 * V_ch3 + D_CH2 * V_ch2 + D_CH * V_ch + D_C * V_c)

y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)

idx = sortperm(z)
plot!(plot2, z[idx], y_pred[idx], lw = 2, label = "model")

display(plot2)