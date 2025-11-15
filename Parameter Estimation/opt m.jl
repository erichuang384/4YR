using CSV, DataFrames
using Clapeyron
using Plots
using Printf, LaTeXStrings, Statistics
using CMAEvolutionStrategy
using Random

# -----------------------------------
# Define models and data files
# -----------------------------------
models = [
    SAFTgammaMie(["Pentane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Octane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Decane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Pentadecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Heptadecane"], idealmodel = WalkerIdeal)

 #   SAFTgammaMie(["Eicosane"], idealmodel = WalkerIdeal)
]

models = [
    SAFTgammaMie(["Pentane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Octane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Decane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Pentadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Heptadecane"], idealmodel = BasicIdeal)

 #   SAFTgammaMie(["Eicosane"], idealmodel = WalkerIdeal)
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
    "Validation Data/Heptadecane DETHERM.csv"
 #   "Training DATA/n-eicosane.csv"
]

# -----------------------------------
# Reduced viscosity calculation for group-contribution model
# -----------------------------------
function reduced_visc(model::EoSModel, P, T, visc)
    visc_CE = IB_CE(model, T)
    s_res = entropy_res(model, P, T)
    s_id  = Clapeyron.entropy(model, P, T) .- s_res
    R     = Clapeyron.Rgas()
    s_red = -s_res / R
    total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])
    m_gc = sum(model.groups.n_groups[1])

    z_term = (-s_res ./ R)./total_sf #.+ log.(-s_res ./ R) ./ total_sf
    
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A
    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A
    n_reduced = (visc .- visc_CE) ./ ((ρ_N .^ (2/3)) .* sqrt.(m .* k_B .* T)) #.* (s_red .^ (2/3))
    return n_reduced, z_term
end

# -----------------------------------
# Precompute per-component datasets and group data
# -----------------------------------
model_names = [m.groups.components[1] for m in models]

data_z_list     = Vector{Vector{Float64}}(undef, length(models))
data_y_list     = Vector{Vector{Float64}}(undef, length(models))
num_CH3_list    = Vector{Float64}(undef, length(models))
num_CH2_list    = Vector{Float64}(undef, length(models))
S_CH3_list      = Vector{Float64}(undef, length(models))
S_CH2_list      = Vector{Float64}(undef, length(models))
sigma_CH3_list  = Vector{Float64}(undef, length(models))
sigma_CH2_list  = Vector{Float64}(undef, length(models))
m_gc_list       = Vector{Float64}(undef, length(models))
total_sf_list   = Vector{Float64}(undef, length(models))
temp_list = Vector{Vector{Float64}}(undef, length(models))
s_id_list = Vector{Vector{Float64}}(undef, length(models))
epsilon_list = Vector{Float64}(undef, length(models))
x_sk_list = Vector{Vector{Float64}}(undef, length(models))
crit_pure_list = Vector{Float64}(undef, length(models))

acentric_list = Vector{Float64}(undef, length(models))

total_points = Ref(0)

for (i, (model, file)) in enumerate(zip(models, data_files))
    try
        exp_data = CSV.read(file, DataFrame)
        P, T, visc_exp = exp_data[:, 1], exp_data[:, 2], exp_data[:, 3]

        results  = reduced_visc.(models[i], P, T, visc_exp)
        n_red    = [r[1] for r in results]
        z_term   = [r[2] for r in results]

        mask = isfinite.(n_red) .& isfinite.(z_term) .& (n_red .> -1)

        data_z_list[i] = z_term[mask]
        #data_y_list[i] = log.(n_red[mask] .+ 1)    # stable ln(1 + n_red)
        data_y_list[i] = log.(n_red[mask] .+ 1.0) 
        # Precompute group data
        groups = model.groups.groups[1]
        @assert groups == ["CH3", "CH2"] "Unexpected groups: $groups"
        num_CH3_list[i] = model.groups.n_groups[1][1]
        num_CH2_list[i] = model.groups.n_groups[1][2]
        S_CH3_list[i] = model.params.shapefactor.values[1]
        S_CH2_list[i] = model.params.shapefactor.values[2]
        sigma_CH3_list[i] = diag(model.params.sigma.values)[1] * 1e10
        sigma_CH2_list[i] = diag(model.params.sigma.values)[2] * 1e10
        m_gc_list[i] = sum(model.groups.n_groups[1])
        total_sf_list[i] = sum(model.params.shapefactor.values .* model.groups.n_groups[1])
        temp_list[i] = T
        crit_point = crit_pure(model)
        crit_pure_list[i] = crit_point[1]
        s_id_list[i] = entropy_ideal.(model, P, T)

        epsilon_list[i] = ϵ_OFE(models[i])
        x_sk_list[i] = x_sk(models[i])
        acentric_list[i] = acentric_factor(model)

        total_points[] += length(data_z_list[i])
    catch err
        @warn "Skipping invalid dataset" file error=err
        data_z_list[i] = Float64[]
        data_y_list[i] = Float64[]
        num_CH3_list[i] = NaN
        num_CH2_list[i] = NaN
        S_CH3_list[i] = NaN
        S_CH2_list[i] = NaN
        sigma_CH3_list[i] = NaN
        sigma_CH2_list[i] = NaN
        m_gc_list[i] = NaN
    end
end

println("Loaded $(total_points[]) valid data points across $(length(models)) datasets.")
if total_points[] == 0
    error("No valid data points loaded — check file paths and data formats.")
end

#... keep your model definition, reduced_visc, and data precomputation blocks unchanged...

println("\nStarting CMA-ES optimization of per-model m parameters...")

# ---------------------------------------------------------------------
# Fixed group-contribution parameters (from your provided optimized set)
# ---------------------------------------------------------------------

A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma_i, D_CH3, D_CH2 = 0.5228313727462552, 0.0020034606250277187, 0.023247265149178208, -1.412098930182651, 0.031223158922453472, -0.019459668181730262, 0.3006410500841957, -0.00012220134519441977, 0.00021878335344363025

# ---------------------------------------------------------------------
# Objective: optimize only m (one value per model)
# ---------------------------------------------------------------------
function sse_m_only(m_vec::AbstractVector{<:Real})
    if length(m_vec) != length(models)
        return 1e20
    end
    A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma_i, D_CH3, D_CH2 = 0.5228313727462552, 0.0020034606250277187, 0.023247265149178208, -1.412098930182651, 0.031223158922453472, -0.019459668181730262, 0.3006410500841957, -0.00012220134519441977, 0.00021878335344363025

    total = 0.0
    for i in 1:length(models)
        z = data_z_list[i]
        y = data_y_list[i]
        if isempty(z)
            continue
        end

        # Precomputed info
        num_CH3   = num_CH3_list[i]
        num_CH2   = num_CH2_list[i]
        S_CH3     = S_CH3_list[i]
        S_CH2     = S_CH2_list[i]
        sigma_CH3 = sigma_CH3_list[i]
        sigma_CH2 = sigma_CH2_list[i]
        T         = temp_list[i]
        T_C       = crit_pure_list[i]
        x_sk_model = x_sk_list[i]

        # Volume-like term
        V = num_CH3 * S_CH3 * sigma_CH3^3 + num_CH2 * S_CH2 * sigma_CH2^3

        # Use SAME DEFINITIONS in objective and plots
        n_g1 = A_vdw_opt(x_sk_model, A_CH3, A_CH2)

        n_g2 = (B_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 +
                B_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) / V^gamma_i

        m_i = m_vec[i]
        n_g3 = (C_CH3 * num_CH3 + C_CH2 * num_CH2).* (1 .+ m_i .* sqrt.(T./ T_C))

        D = (D_CH3 * num_CH3 * S_CH3 * sigma_CH3^3 +
             D_CH2 * num_CH2 * S_CH2 * sigma_CH2^3)

        y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)

        if any(!isfinite, y_pred)
            total += 1e20
            continue
        end

        # normalized SSE on eta = exp(y) - 1
        eta = (exp.(y).- 1)
        eta_pred = exp.(y_pred).- 1
        total += sum(((eta.- eta_pred)./ eta).^ 2) / length(y)
    end

    return isfinite(total) ? total : 1e20
end

# ---------------------------------------------------------------------
# CMA-ES setup: initial guess for m per model
# Use original correlation as initial m_i
# ---------------------------------------------------------------------
m0 = [0.379642 + 1.54226 * acentric_list[i] - 0.26992 * acentric_list[i]^2
      for i in 1:length(models)]

σ0 = 0.01
seed = 42
Random.seed!(seed)
stagnation_iters = 4000
iter_counter = Ref(0)

result = minimize(
    sse_m_only,
    m0,
    σ0;
    seed = seed,
    verbosity = 2,
    stagnation = stagnation_iters,
    maxiter = 10000,
    ftol = 1e-10,
    callback = (opt, x, fx, ranks) -> begin
        iter_counter[] += 1
        if iter_counter[] % 50 == 0
            try
                println(@sprintf("Iter %d: fmin=%.6e best=%s",
                                 iter_counter[], minimum(fx), string(xbest(opt))))
            catch
                println(@sprintf("Iter %d: callback invoked (no xbest/fmin available)", iter_counter[]))
            end
        end
    end
)

m_opt = xbest(result)
final_sse = fbest(result)

println("\n✅ CMA-ES optimization (m only) successful!")
@printf "Final SSE = %.8e\n" final_sse
@printf "Optimized m per model:\n"
for (i, name) in enumerate(model_names)
    @printf "  %-12s m = %.8f\n" name m_opt[i]
end

# ---------------------------------------------------------------------
# Plot 1: All components — model vs data with optimized m
# ---------------------------------------------------------------------
plot1 = plot(
    grid = false,
    xlabel = L"z_{\mathrm{term}}",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Dimensionless Viscosity Collapse — CMA-ES (m-only optimization)",
    legend = :bottomright,
    lw = 3
)

# Scatter all experimental datapoints
for i in eachindex(models)
    z = data_z_list[i]
    y = data_y_list[i]
    if !isempty(z)
        scatter!(plot1, z, y, label=false, marker=:auto)
    end
end

# Overlay model prediction curves for each component (consistent with objective)
for i in eachindex(models)
    z = data_z_list[i]
    if isempty(z)
        continue
    end

    num_CH3   = num_CH3_list[i]
    num_CH2   = num_CH2_list[i]
    S_CH3     = S_CH3_list[i]
    S_CH2     = S_CH2_list[i]
    sigma_CH3 = sigma_CH3_list[i]
    sigma_CH2 = sigma_CH2_list[i]
    T         = temp_list[i]
    T_C       = crit_pure_list[i]
    x_sk_model = x_sk_list[i]

    V = num_CH3 * S_CH3 * sigma_CH3^3 + num_CH2 * S_CH2 * sigma_CH2^3

    n_g1 = A_vdw_opt(x_sk_model, A_CH3, A_CH2)

    n_g2 = (B_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 +
            B_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) / V^gamma_fix

    m_i = m_opt[i]
    n_g3 = (C_CH3 * num_CH3 + C_CH2 * num_CH2).* (1.+ sqrt.(m_i.* T./ T_C))

    D = (D_CH3 * num_CH3 * S_CH3 * sigma_CH3^3 +
         D_CH2 * num_CH2 * S_CH2 * sigma_CH2^3)

    y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)

    idx = sortperm(z)
    plot!(
        plot1,
        z[idx], y_pred[idx],
        lw = 2,
        label = "$(model_names[i]) model"
    )
end

display(plot1)

# ---------------------------------------------------------------------
# Plot 2: Single component (index = 6) — model vs data with optimized m
# ---------------------------------------------------------------------
index = 6
plot2 = plot(
    grid = false,
    xlabel = L"z_{\mathrm{term}}",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Dimensionless Viscosity Collapse — CMA-ES (m-only optimization)",
    legend = :bottomright,
    lw = 3
)

# Scatter experimental data
z = data_z_list[index]
y = data_y_list[index]
if !isempty(z)
    scatter!(plot2, z, y, label=false, marker=:auto)
end

# Overlay model prediction (consistent with objective)
num_CH3   = num_CH3_list[index]
num_CH2   = num_CH2_list[index]
S_CH3     = S_CH3_list[index]
S_CH2     = S_CH2_list[index]
sigma_CH3 = sigma_CH3_list[index]
sigma_CH2 = sigma_CH2_list[index]
T         = temp_list[index]
T_C       = crit_pure_list[index]
x_sk_model = x_sk_list[index]

V = num_CH3 * S_CH3 * sigma_CH3^3 + num_CH2 * S_CH2 * sigma_CH2^3

n_g1 = A_vdw_opt(x_sk_model, A_CH3, A_CH2)

n_g2 = (B_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 +
        B_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) / V^gamma_fix

m_i = m_opt[index]
n_g3 = (C_CH3 * num_CH3 + C_CH2 * num_CH2).* (1.+ sqrt.(m_i.* T./ T_C))

D = (D_CH3 * num_CH3 * S_CH3 * sigma_CH3^3 +
     D_CH2 * num_CH2 * S_CH2 * sigma_CH2^3)

y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)

idx = sortperm(z)
plot!(
    plot2,
    z[idx], y_pred[idx],
    lw = 2,
    label = false
)

display(plot2)