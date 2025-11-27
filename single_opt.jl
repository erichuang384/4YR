using CSV, DataFrames
using Clapeyron
using Plots
using Printf, LaTeXStrings, Statistics
using CMAEvolutionStrategy
using Random

# -----------------------------------
# Models and data (branched alkanes)
# -----------------------------------
models = [
    SAFTgammaMie(["squalane"])
]


data_files = [
    "Training DATA/Branched Alkane/squalane.csv"
]
# -----------------------------------
# Reduced viscosity calculation for group-contribution model
# -----------------------------------
function reduced_visc(model::EoSModel, P, T, visc)
    visc_CE = IB_CE(model, T)
    s_res   = entropy_res(model, P, T)
    R       = Clapeyron.Rgas()
    total_sf = sum(model.params.shapefactor.values.* model.groups.n_groups[1])

    # z-term
    z_term = (-s_res./ R)./ total_sf

    # density and molecular mass for reduced viscosity
    N_A     = Clapeyron.N_A
    k_B     = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N     = ρ_molar.* N_A
    Mw      = Clapeyron.molecular_weight(model)
    m       = Mw / N_A

    n_reduced = (visc .- visc_CE)./ ((ρ_N.^ (2/3)).* sqrt.(m.* k_B.* T))
    return n_reduced, z_term
end

# -----------------------------------
# Precompute datasets and group data
# -----------------------------------

model_names    = [m.groups.components[1] for m in models]
data_z_list    = Vector{Vector{Float64}}(undef, length(models))
data_y_list    = Vector{Vector{Float64}}(undef, length(models))
data_T_list    = Vector{Vector{Float64}}(undef, length(models))  # masked T aligned with z,y
crit_pure_list = Vector{Float64}(undef, length(models))
crit_pres_list =    Vector{Float64}(undef, length(models))
res_crit_entr_list = Vector{Float64}(undef, length(models))
reduced_crit_temp = Vector{Float64}(undef, length(models))

x_sk_list      = Vector{Vector{Float64}}(undef, length(models))
Mw_list        = Vector{Float64}(undef, length(models))
m_gc_list       = Vector{Float64}(undef, length(models))

# Group counts and parameters
ch3_count_list    = Vector{Float64}(undef, length(models))
ch2_count_list    = Vector{Float64}(undef, length(models))
ch_count_list     = Vector{Float64}(undef, length(models))
c_count_list     = Vector{Float64}(undef, length(models))
sigma_ch3_list    = Vector{Float64}(undef, length(models))
sigma_ch2_list    = Vector{Float64}(undef, length(models))
sigma_ch_list     = Vector{Float64}(undef, length(models))
sigma_c_list     = Vector{Float64}(undef, length(models))
S_ch3_model_list  = Vector{Float64}(undef, length(models))
S_ch2_model_list  = Vector{Float64}(undef, length(models))
S_ch_model_list   = Vector{Float64}(undef, length(models))
S_c_model_list   = Vector{Float64}(undef, length(models))
acentric_factor_list =  Vector{Float64}(undef, length(models))

total_points = Ref(0)

for (i, (model, file)) in enumerate(zip(models, data_files))
    try
        exp_data = CSV.read(file, DataFrame)
        P, T, visc_exp = exp_data[:, 1], exp_data[:, 2], exp_data[:, 3]

        # Compute reduced viscosity and z-term
        results  = reduced_visc.(models[i], P, T, visc_exp)
        n_red    = [r[1] for r in results]
        z_term   = [r[2] for r in results]

        mask = isfinite.(n_red).& isfinite.(z_term).& (n_red.> -1)

        data_z_list[i] = z_term[mask]
        data_y_list[i] = log.(n_red[mask] .+ 1.0 )
        data_T_list[i] = T[mask]  # store masked T aligned with z,y

        # Critical temperature
        crit_point        = crit_pure(model)
        crit_pure_list[i] = crit_point[1]
        crit_pres_list[i] = crit_point[2]
        res_crit_entr_list[i] = -entropy_res(model, crit_point[2], crit_point[1])/Rgas()
        reduced_crit_temp[i] = crit_point[1]/ϵ_OFE(model)

        # x_sk and Mw
        x_sk_list[i] = x_sk(models[i])
        Mw_list[i]   = Clapeyron.molecular_weight(model)

        # Extract group data via direct findfirst
        groups = model.groups.groups[1]             # group names
        ng     = model.groups.n_groups[1]           # counts per group
        Svals  = model.params.shapefactor.values    # shapefactors aligned with groups
        σdiag  = diag(model.params.sigma.values)    # sigma per group (m), diagonal

        i_ch3 = findfirst(==("CH3"), groups)
        i_ch2 = findfirst(==("CH2"), groups)
        i_ch  = findfirst(==("CH"),  groups)
        i_c   = findfirst(==("C"),  groups)

        ch3_count_list[i]    = i_ch3 === nothing ? 0.0 : ng[i_ch3]
        S_ch3_model_list[i]  = i_ch3 === nothing ? 0.0 : Svals[i_ch3]
        sigma_ch3_list[i]    = i_ch3 === nothing ? 0.0 : σdiag[i_ch3]*1e10  # Å

        ch2_count_list[i]    = i_ch2 === nothing ? 0.0 : ng[i_ch2]
        S_ch2_model_list[i]  = i_ch2 === nothing ? 0.0 : Svals[i_ch2]
        sigma_ch2_list[i]    = i_ch2 === nothing ? 0.0 : σdiag[i_ch2]*1e10  # Å

        ch_count_list[i]     = i_ch  === nothing ? 0.0 : ng[i_ch]
        S_ch_model_list[i]   = i_ch  === nothing ? 0.0 : Svals[i_ch]
        sigma_ch_list[i]     = i_ch  === nothing ? 0.0 : σdiag[i_ch]*1e10   # Å

        c_count_list[i]     = i_c  === nothing ? 0.0 : ng[i_c]
        S_c_model_list[i]   = i_c  === nothing ? 0.0 : Svals[i_c]
        sigma_c_list[i]     = i_c  === nothing ? 0.0 : σdiag[i_c]*1e10   # Å
        acentric_factor_list[i] = acentric_factor(model)


        m_gc_list[i]        = sum(ng.* Svals)

        total_points[] += length(data_z_list[i])
    catch err
        @warn "Skipping invalid dataset" file error=err
        data_z_list[i]     = Float64[]
        data_y_list[i]     = Float64[]
        data_T_list[i]     = Float64[]
        crit_pure_list[i]  = NaN
        x_sk_list[i]       = Float64[]
        Mw_list[i]         = NaN
        ch3_count_list[i]  = NaN
        ch2_count_list[i]  = NaN
        ch_count_list[i]   = NaN
        c_count_list[i]   = NaN
        S_ch3_model_list[i]= NaN
        S_ch2_model_list[i]= NaN
        S_ch_model_list[i] = NaN
        S_c_model_list[i] = NaN
        sigma_ch3_list[i]  = NaN
        sigma_ch2_list[i]  = NaN
        sigma_ch_list[i]   = NaN
        sigma_c_list[i]   = NaN
    end
end

println("Loaded $(total_points[]) valid data points across $(length(models)) datasets.")
if total_points[] == 0
    error("No valid data points loaded — check file paths and data formats.")
end

# ------------------------------------------------------------
# Fixed constants from your 12-parameter optimization (CH3/CH2 set)
# [A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, gamma, D_CH3, D_CH2, m1, m2, m3]
# ------------------------------------------------------------

x0 = [0.3, 0.5, 0.5, 0.001, 1.0]
x0 = [-1.7959349402713636, 4.359772186971961, -0.8308696445362812, 0.29008728624727087, 1.2400951096804953]
println("Initial CH guess:")


function sse_all(params::AbstractVector{<:Real})
    if length(params) != 5
        return 1e20
    end
    #A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, A_CH, B_CH, C_CH, gamma_i, D_i, m_1, m_2, m_3 = params
    A, B, C, D, m = params
    total = 0.0

    for i in 1:length(models)
        z = data_z_list[i]
        y = data_y_list[i]
        T = data_T_list[i]
        if isempty(z); continue; end


        Tc  = crit_pure_list[i]

        # Model terms
        n_g1 = A

        n_g2 = B

        n_g3 = C * (1 .+ m *(T ./ Tc) .^ 1.0)

        y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)
        if any(!isfinite, exp.(y_pred))
            total += 1e10
            continue
        end

        # normalized SSE on eta = exp(y) - 1
        η      = exp.(y) .- 1.0
        η_pred = exp.(y_pred) .- 1.0
        total += (sum(((η.- η_pred)./ η).^ 2) / length(y) ) 
    end

    return isfinite(total) ? total : 1e20
end

# ------------------------------------------------------------
# Run CMA-ES to optimize CH parameters
# ------------------------------------------------------------
println("\nStarting CMA-ES optimization of CH parameters (A_CH, B_CH, C_CH, D_CH) on branched alkanes...")

σ0 = 0.01
seed = 42
Random.seed!(seed)
stagnation_iters = 10000
iter_counter = Ref(0)

result = minimize(
    sse_all,
    x0,
    σ0;
    seed = seed,
    verbosity = 2,
    stagnation = stagnation_iters,
    maxiter = 100000,
    ftol = 1e-30,
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
final_sse  = fbest(result)
A_CH, B_CH, C_CH, D_CH = params_opt

println("\n✅ CMA-ES CH-only optimization successful!")
@printf "Optimized CH parameters:\n"
@printf "  A_CH = %.10f\n" A_CH
@printf "  B_CH = %.10f\n" B_CH
@printf "  C_CH = %.10f\n" C_CH
@printf "  D_CH = %.10f\n" D_CH
@printf "Final SSE = %.8e\n" final_sse
