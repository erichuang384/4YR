using CSV, DataFrames
using Clapeyron
using Plots
using Printf, LaTeXStrings, Statistics
using CMAEvolutionStrategy
using Random
include("dev partial.jl")
# -----------------------------------
# Define models and data files
# -----------------------------------
# models = [
#     SAFTgammaMie(["Pentane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Hexane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Octane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Decane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Dodecane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Tridecane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Pentadecane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Hexadecane"], idealmodel = WalkerIdeal),
#     SAFTgammaMie(["Heptadecane"], idealmodel = WalkerIdeal)

#  #   SAFTgammaMie(["Eicosane"], idealmodel = WalkerIdeal)
# ]

models = [
    SAFTgammaMie(["Butane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Pentane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Octane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Decane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = BasicIdeal),
   # SAFTgammaMie(["Pentadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Heptadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Eicosane"], idealmodel = BasicIdeal)
]

crit_point_list = crit_pure.(models)
crit_entropy_list = zeros(length(models))
for i in 1:length(models)
    crit_entropy_list[i] = entropy_res(models[i], crit_point_list[i][2],crit_point_list[i][1])
end
res_crit_entropy_list = -crit_entropy_list ./ Rgas()


data_files = [
    "Training DATA/Butane DETHERM.csv",
    "Training DATA/Pentane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Validation Data/Tridecane DETHERM.csv",
    "Validation Data/Pentadecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv",
    "Validation Data/Heptadecane DETHERM.csv",
    "Training DATA/n-eicosane.csv"
]

# -----------------------------------
# Reduced viscosity calculation for group-contribution model
# -----------------------------------
function reduced_visc(model::EoSModel, P, T, visc)
    visc_CE = IB_CE(model, T)
    s_res = entropy_res(model, P, T)
    #s_id  = Clapeyron.entropy(model, P, T) .- s_res
    R     = Clapeyron.Rgas()
    #s_red = -s_res / R
    total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])
    #m_gc = sum(model.groups.n_groups[1])

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
pres_list = Vector{Vector{Float64}}(undef, length(models))

gam_exp_list = Vector{Vector{Float64}}(undef, length(models))
crit_pres_list = Vector{Float64}(undef, length(models))

s_id_list = Vector{Vector{Float64}}(undef, length(models))
epsilon_list = Vector{Float64}(undef, length(models))
x_sk_list = Vector{Vector{Float64}}(undef, length(models))
crit_pure_list = Vector{Float64}(undef, length(models))

acentric_list = Vector{Float64}(undef, length(models))
Mw_list = Vector{Float64}(undef, length(models))

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
        pres_list[i] = P
        gam_exp_list[i] = gamma_exponent.(model,P,T)
        crit_point = crit_pure(model)
        crit_pure_list[i] = crit_point[1]
        crit_pres_list[i] = crit_point[2]
        s_id_list[i] = entropy_ideal.(model, P, T)

        epsilon_list[i] = ϵ_OFE(models[i])
        x_sk_list[i] = x_sk(models[i])
        acentric_list[i] = acentric_factor(model)
        Mw_list[i] = Clapeyron.molecular_weight(model)

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

# -----------------------------------
# CMA-ES objective: optimize group-contribution params [A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma, D_i]
# -----------------------------------
function sse_group_contrib(params::AbstractVector{<:Real})
    if length(params) != 12
        return 1e20
    end
    #A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma, D_CH3, D_CH2 = params
    A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma, D_CH3, D_CH2, m_1, m_2, m_3 = params
    #A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma, D_i, m_1, m_2, m_3 = params
    total = 0.0
    for i in 1:length(models)
        z = data_z_list[i]
        y = data_y_list[i]
        if isempty(z)
            continue
        end
        num_CH3 = num_CH3_list[i]
        num_CH2 = num_CH2_list[i]
        S_CH3 = S_CH3_list[i]
        S_CH2 = S_CH2_list[i]
        sigma_CH3 = sigma_CH3_list[i]
        sigma_CH2 = sigma_CH2_list[i]
        m_gc = m_gc_list[i]
        tot_sf = total_sf_list[i]
        T = temp_list[i]
        P = pres_list[i]
        gamma_exp = gam_exp_list[i]
        T_C = crit_pure_list[i]
        P_C = crit_pres_list[i]

        epsilon = epsilon_list[i]
        x_sk_model = x_sk_list[i]
        s_id = s_id_list[i]
        acentric_fact = acentric_list[i]
        Mw_model  = Mw_list[i]
        

        V = num_CH3 * S_CH3 * sigma_CH3^3 + num_CH2 * S_CH2 * sigma_CH2^3
        #n_g1 = A_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 + A_CH2 * S_CH2 * sigma_CH2^3 * num_CH2
        #n_g1 = (A_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 + A_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) #/ (num_CH2+num_CH3)
        n_g1 = A_vdw_opt(models[i],A_CH3,A_CH2,0.0,0.0)
        n_g2 = (B_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 + B_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) / V^gamma
        #n_g2 = A_vdw(x_sk_model, B_CH3, B_CH2) #/ V ^ gamma

        #m = 0.379642 + 1.54226 * acentric_fact - 0.26992*acentric_fact^2
        #m_T = m_1 + m_2 * Mw_model + m_3 * Mw_model^2
        #gam_exp = gamma_exponent.(models[i], P, T) .- gamma_exponent(models[i],P_C,T_C)
        m_T = m_1 .+ m_2 .* gamma_exp .+ m_3 .* gamma_exp .^ 2
        
        #m_T = m_1 + m_2 * m_gc + m_3 * m_gc ^ 2
        #m = m_1 + m_2 * acentric_fact + m_3 * acentric_fact^2

        n_g3 = (C_CH3 .* num_CH3 .+ C_CH2 .* num_CH2) .*  (1 .+ m_T .* sqrt.(T./T_C))
        #n_g3 = (C_CH3 .* num_CH3 .+ C_CH2 .* num_CH2) .* s_id
        #n_g3 = A_vdw_opt(x_sk_model,C_CH3, C_CH2) .* s_id
        #n_g3 = (log.( A_vdw_opt(x_sk_model, C_CH3,C_CH2).* T ./ T_C))
        #n_g3 = A_vdw(x_sk_model, C_CH3, C_CH2) .* (1 .+ log.(T))

        #D = D_i .* m_gc
        D = (D_CH3 .* num_CH3 .* S_CH3 * sigma_CH3^3 .+ D_CH2 .* num_CH2 .* S_CH2 * sigma_CH2^3) #.* (1 .+ log.(T))
        #D  = A_vdw(x_sk_model,D_CH3,D_CH2)

        y_pred = n_g1 .+ n_g2 .* z .+ n_g3 .* (z .^ 2) .+ D .* (z .^ 3)
        if any(!isfinite, y_pred)
            total += 1e20
            continue
        end
        # normalized SSE
        eta = (exp.(y) .- 1)
        eta_pred = exp.(y_pred) .-1
        total += sum(((eta .- eta_pred) ./ eta) .^ 2) / length(y)
        #total += sum(((y .- y_pred) ./ y) .^ 2) / length(y)
    end
    return isfinite(total) ? total : 1e20
end



println("\nStarting CMA-ES optimization of group-contribution parameters...")

# Initial guess and bounds
x0 = [0.07499014575254812, 0.022952089265885545, -0.07399881535579876, -1.1555094778528896, 0.03046889724552433, -0.0015829542219616076, 0.33134385697591257, 0.00020404016733679533, -0.89664645425576, 8.430884960547536, 5.5927245874714435]
x0 = [0.19559028955933497, 0.01891278652141742, -0.0676141053733902, -1.2655624213727277, 0.04579084462791991, -0.004667007395279716, 0.3805297833158242, 0.00017994450415668582, 0.00013889230918045332, -1.1230042615433207, 3.3689002130697854, 0.7584844975125721]
# For no sf on z
x0 = [-0.06046671299667853, 0.3737836229207754, -7.098706802783836e-5, -0.6908323087652994, 0.27971954480593647, 0.00011222552630781913, 0.9988512642395981, 1.539793408904002e-5, -1.954697661169505e-6, -17.706835766026753, 32.73603031929364, 23.688913248745667]
# For whole z division
x0 = [0.6145360292355571, -8.559333420885814, 0.10300016800791054, -2.125091141241104, 23.487200356606454, -0.09629694834867396, 1.4888962872895215, -0.00018615415970691732, 0.0003568669321560236, -0.6825669125011607, 12.533744035717795, 15.510758387701088]
# For mgc for tau
x0 = [-0.030412555394213076, 0.0003759524822874654, -0.0788585031729372, -1.0894087686841185, 2.3172430099106588e-6, -0.0002924330416947319, -0.6617771433184333, 0.00021759950159757008, 0.00011712305344327157, -0.6791089104679724, 0.08324012291665168, 0.002834878605430142]

lower = nothing #[-1.0, -5.0, -5.0, -5.0, -5.0, -5.0, -1.0, -10.0]
upper = nothing #[1.0, 5.0,   5.0, 5.0, 5.0,   5.0,   1.0, 10.0]

# CMA-ES hyperparameters
σ0 = 0.01
seed = 42
Random.seed!(seed)
#λ = 4 + floor(Int, 3 * log(length(x0)))
stagnation_iters = 5000
iter_counter = Ref(0)

result = minimize(
    sse_group_contrib,
    x0,
    σ0;
    lower = lower,
    upper = upper,
    seed = seed,
    verbosity = 2,
    #popsize = λ,
    stagnation = stagnation_iters,
    maxiter = 30000,
    ftol = 1e-15,
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

params_opt = xbest(result)
final_sse = fbest(result)

# Unpack optimized parameters
A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma_opt, D_i_opt = params_opt

println("\n✅ CMA-ES optimization successful!")
@printf "Group coefficients:\n"
@printf "  A_CH3 = %.8f\n" A_CH3
@printf "  B_CH3 = %.8f\n" B_CH3
@printf "  C_CH3 = %.8f\n" C_CH3
@printf "  A_CH2 = %.8f\n" A_CH2
@printf "  B_CH2 = %.8f\n" B_CH2
@printf "  C_CH2 = %.8f\n" C_CH2
@printf "  gamma = %.8f\n" gamma_opt
@printf "  D_i   = %.8f\n" D_i_opt
@printf "Final SSE = %.8e\n" final_sse
# -----------------------------------
# Corrected plotting section — now matches the optimization model
# -----------------------------------

plot1 = plot(
    grid = false,
    xlabel = L"z_{\mathrm{term}}",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Dimensionless Viscosity Collapse — CMA-ES group-contribution",
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

# Overlay model prediction curves for each component
for i in eachindex(models)

    z = data_z_list[i]
    if isempty(z)
        continue
    end

    # --- Extract precomputed info ---
    num_CH3   = num_CH3_list[i]
    num_CH2   = num_CH2_list[i]
    S_CH3     = S_CH3_list[i]
    S_CH2     = S_CH2_list[i]
    sigma_CH3 = sigma_CH3_list[i]
    sigma_CH2 = sigma_CH2_list[i]
    T         = temp_list[i]
    epsilon   = epsilon_list[i]
    x_sk_model = x_sk_list[i]

    # Volume-like term from objective function
    V = num_CH3 * S_CH3 * sigma_CH3^3 + num_CH2 * S_CH2 * sigma_CH2^3

    # --- Use EXACT SAME DEFINITIONS as sse_group_contrib() ---
    n_g1 = A_vdw(x_sk_model, A_CH3, A_CH2)

    n_g2 = (B_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 +
            B_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) / V^gamma_opt

    n_g3 = (C_CH3 * num_CH3 .+ C_CH2 * num_CH2) .* (log.(T))

    D = (D_CH3 * num_CH3 * S_CH3 * sigma_CH3^3 +
         D_CH2 * num_CH2 * S_CH2 * sigma_CH2^3)

    # Predicted log(η_res⁺ + 1)
    y_pred = n_g1 .+ n_g2 .* z .+ n_g3 .* (z .^ 2) .+ D .* (z .^ 3)

    # Plot smooth curve
    idx = sortperm(z)
    plot!(
        plot1,
        z[idx], y_pred[idx],
        lw = 2,
        label = "$(model_names[i]) model"
    )
end

display(plot1)


index = 6
plot2 = plot(
    grid = false,
    xlabel = L"z_{\mathrm{term}}",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Dimensionless Viscosity Collapse — CMA-ES group-contribution",
    legend = :bottomright,
    lw = 3
)

# Scatter all experimental datapoints

    z = data_z_list[index]
    y = data_y_list[index]
    if !isempty(z)
        scatter!(plot2, z, y, label=false, marker=:auto)
    end


# Overlay model prediction curves for each component


    z = data_z_list[index]


    # --- Extract precomputed info ---
    num_CH3   = num_CH3_list[index]
    num_CH2   = num_CH2_list[index]
    S_CH3     = S_CH3_list[index]
    S_CH2     = S_CH2_list[index]
    sigma_CH3 = sigma_CH3_list[index]
    sigma_CH2 = sigma_CH2_list[index]
    T         = temp_list[index]
    epsilon   = epsilon_list[index]
    x_sk_model = x_sk_list[index]

    # Volume-like term from objective function
    V = num_CH3 * S_CH3 * sigma_CH3^3 + num_CH2 * S_CH2 * sigma_CH2^3

    # --- Use EXACT SAME DEFINITIONS as sse_group_contrib() ---
    n_g1 = A_vdw(x_sk_model, A_CH3, A_CH2)

    n_g2 = (B_CH3 * S_CH3 * sigma_CH3^3 * num_CH3 +
            B_CH2 * S_CH2 * sigma_CH2^3 * num_CH2) / V^gamma_opt

    n_g3 = (C_CH3 * num_CH3 .+ C_CH2 * num_CH2) .* (log.(T))

    D = (D_CH3 * num_CH3 * S_CH3 * sigma_CH3^3 +
         D_CH2 * num_CH2 * S_CH2 * sigma_CH2^3)

    # Predicted log(η_res⁺ + 1)
    y_pred = n_g1 .+ n_g2 .* z .+ n_g3 .* (z .^ 2) .+ D .* (z .^ 3)

    # Plot smooth curve
    idx = sortperm(z)
    plot!(
        plot2,
        z[idx], y_pred[idx],
        lw = 2,
        label = false
    )


display(plot2)
