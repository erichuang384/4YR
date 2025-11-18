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
     SAFTgammaMie(["2,2,4-trimethylpentane"]),
     SAFTgammaMie(["heptamethylnonane"]),
     SAFTgammaMie(["2,2,4-trimethylhexane"])
#= #    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
  #  SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["2-methylbutane"]),
    SAFTgammaMie(["2-methylpentane"]),
#    SAFTgammaMie(["9-octylheptadecane"]), # very questionable data
    SAFTgammaMie(["squalane"]),
    SAFTgammaMie(["3,5-dimethylheptane"]),
    SAFTgammaMie(["2-methyloctane"]),
    SAFTgammaMie(["4-methyloctane"]),
    SAFTgammaMie(["3,6-dimethyloctane"]),
    SAFTgammaMie(["2-methylnonane"]),
    SAFTgammaMie(["2-methyldecane"]),
    SAFTgammaMie(["3-methylundecane"]),
    SAFTgammaMie(["2-methylpentadecane"]),
    SAFTgammaMie(["7-methylhexadecane"]), =#

]

data_files = [
    "Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,2,4,4,6,8,8-heptamethylnonane.csv",
    "Training DATA/Branched Alkane/2,2,4-trimethylhexane.csv"
#    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
 #   "Training DATA/Branched Alkane/2-methylpropane.csv",
#=     "Training DATA/Branched Alkane/2-methylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpentane.csv",
#    "Training DATA/Branched Alkane/9-octylheptadecane.csv",
    "Training DATA/Branched Alkane/squalane.csv",
    "Training DATA/Branched Alkane/3,5-dimethylheptane.csv",
    "Training DATA/Branched Alkane/2-methyloctane.csv",
    "Training DATA/Branched Alkane/4-methyloctane.csv",
    "Training DATA/Branched Alkane/3,6-dimethyloctane.csv",
    "Training DATA/Branched Alkane/2-methylnonane.csv",
    "Training DATA/Branched Alkane/2-methyldecane.csv",
    "Training DATA/Branched Alkane/3-methylundecane.csv",
    "Training DATA/Branched Alkane/2-methylpentadecane.csv",
    "Training DATA/Branched Alkane/7-methylhexadecane.csv" =#
    
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
    z_term = (-s_res./ R)#./ total_sf

    # density and molecular mass for reduced viscosity
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
model_names    = [m.groups.components[1] for m in models]
data_z_list    = Vector{Vector{Float64}}(undef, length(models))
data_y_list    = Vector{Vector{Float64}}(undef, length(models))
data_T_list    = Vector{Vector{Float64}}(undef, length(models))  # masked T aligned with z,y
crit_pure_list = Vector{Float64}(undef, length(models))
x_sk_list      = Vector{Vector{Float64}}(undef, length(models))
Mw_list        = Vector{Float64}(undef, length(models))

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
        data_y_list[i] = log.(n_red[mask].+ 1.0)
        data_T_list[i] = T[mask]  # store masked T aligned with z,y

        # Critical temperature
        crit_point        = crit_pure(model)
        crit_pure_list[i] = crit_point[1]

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
        i_c  = findfirst(==("C"),  groups)

        ch3_count_list[i]    = i_ch3 === nothing ? 0.0 : ng[i_ch3]
        S_ch3_model_list[i]  = i_ch3 === nothing ? 0.0 : Svals[i_ch3]
        sigma_ch3_list[i]    = i_ch3 === nothing ? 0.0 : σdiag[i_ch3]*1e10  # Å

        ch2_count_list[i]    = i_ch2 === nothing ? 0.0 : ng[i_ch2]
        S_ch2_model_list[i]  = i_ch2 === nothing ? 0.0 : Svals[i_ch2]
        sigma_ch2_list[i]    = i_ch2 === nothing ? 0.0 : σdiag[i_ch2]*1e10  # Å

        ch_count_list[i]     = i_ch  === nothing ? 0.0 : ng[i_ch]
        S_ch_model_list[i]   = i_ch  === nothing ? 0.0 : Svals[i_ch]
        sigma_ch_list[i]     = i_ch  === nothing ? 0.0 : σdiag[i_ch]*1e10   # Å

        c_count_list[i]     = i_ch  === nothing ? 0.0 : ng[i_c]
        S_c_model_list[i]   = i_ch  === nothing ? 0.0 : Svals[i_c]
        sigma_c_list[i]     = i_ch  === nothing ? 0.0 : σdiag[i_c]*1e10   # Å

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
        S_ch_model_list[i] = 
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
A_CH3_0, B_CH3_0, C_CH3_0,
A_CH2_0, B_CH2_0, C_CH2_0,
A_CH_0, B_CH_0, C_CH_0,
gamma_fix, D_CH3_0, D_CH2_0, D_CH_0,
m1_fix, m2_fix, m3_fix = (0.07499014575254812, 0.022952089265885545, -0.07399881535579876,
 -1.1555094778528896, 0.03046889724552433, -0.0015829542219616076,
 -5.277536540878634, 0.051736443260454255, 0.07203219370665288,
  0.33134385697591257, 0.00020404016733679533, 0.0001216378536253241, 0.0007019909860755935, -0.89664645425576, 8.430884960547536, 5.5927245874714435)

  # For no sf division on z
A_CH3_0, B_CH3_0, C_CH3_0,
A_CH2_0, B_CH2_0, C_CH2_0,
A_CH_0, B_CH_0, C_CH_0,
gamma_fix, D_CH3_0, D_CH2_0, D_CH_0,
m1_fix, m2_fix, m3_fix = (-0.07020613988459963, 0.4526929704410887, -1.6734960099036927e-5,
 -0.6976576704156878, 0.3681270045188838, 9.30920589170978e-5,
 8.130289974449004, -2.197224469355373, -0.00015921723744090146,
  1.040106781887901, 
  1.5674091869700848e-5, -1.999341455459331e-6, -6.147413637510999e-5, -20.2637266914618, 38.55028392360753, 26.27614884148527)


A_CH3_0, B_CH3_0, C_CH3_0,
A_CH2_0, B_CH2_0, C_CH2_0,
gamma_fix, D_i,
m1_fix, m2_fix, m3_fix = (0.1996694035404919, 0.013163399618677435, -0.0395093022663388, -1.1712339187550833, 0.025427903131912948, -0.006610385377581092, 0.2946194908647525, 0.003918383017106727, -1.3580945763938599, 4.958547241103466, -0.662141126516273)

# ------------------------------------------------------------
# Initial guess for CH parameters [A_CH, B_CH, C_CH, D_CH]
# (neutral average from CH3/CH2 constants)
# ------------------------------------------------------------

x0 = [-9.385800472091962, 0.02628502956068002, 0.019599739052107427, 0.0]
println("Initial C guess:")
#@printf "  A_CH = %.10f, B_CH = %.10f, C_CH = %.10f, D_CH = %.10f\n" x0...

# ------------------------------------------------------------
# Objective: optimize CH params only (A_CH, B_CH, C_CH, D_CH)
# CH shapefactor S_ch is taken as model constant (S_ch_model_list), not optimized
# Additive contributions from CH3 + CH2 + CH for n_g2, n_g3, D
# ------------------------------------------------------------
function sse_C_only(params::AbstractVector{<:Real})
    if length(params) != 4
        return 1e20
    end
    A_C, B_C, C_C, D_C = params
    #A_CH, B_CH, C_CH = params
    total = 0.0

    for i in 1:length(models)
        z = data_z_list[i]
        y = data_y_list[i]
        T = data_T_list[i]
        if isempty(z); continue; end

        # Precomputed terms (named ch, ch2, ch3)
        ch3  = ch3_count_list[i]
        ch2  = ch2_count_list[i]
        ch   = ch_count_list[i]
        c   = c_count_list[i]

        S_ch3 = S_ch3_model_list[i]
        S_ch2 = S_ch2_model_list[i]
        S_ch  = S_ch_model_list[i]    # constant from model
        S_c  = S_c_model_list[i]

        sigma_ch3 = sigma_ch3_list[i]^3
        sigma_ch2 = sigma_ch2_list[i]^3
        sigma_ch  = sigma_ch_list[i]^3
        sigma_c  = sigma_c_list[i]^3

        Tc  = crit_pure_list[i]
        xsk = x_sk_list[i]
        Mw  = Mw_list[i]

        # Volumes
        V_ch3 = ch3 * S_ch3 * sigma_ch3
        V_ch2 = ch2 * S_ch2 * sigma_ch2
        V_ch  = ch  * S_ch  * sigma_ch
        V_c  = c  * S_c  * sigma_c
        V_tot = V_ch3 + V_ch2 + V_ch + V_c
        if !(isfinite(V_tot)) || V_tot <= 0
            total += 1e20
            continue
        end

        # Model terms
        n_g1 = A_vdw_opt(models[i], A_CH3_0, A_CH2_0,A_CH_0, A_C)
        #n_g1       = n_g1_fixed + A_CH * (V_ch / V_tot)

        n_g2= ((B_CH3_0 * V_ch3) + (B_CH2_0 * V_ch2) + (B_CH_0 * V_ch)+ (B_C * V_c))/V_tot^gamma_fix
        #n_g2     = n_g2_num / V_tot^gamma_fix

        #acentric_fact = acentric_factor(models[i])
        #m_i  = m1_fix + m2_fix*acentric_fact + m3_fix*acentric_fact^2

        m_i  = m1_fix + m2_fix*Mw + m3_fix*Mw^2

        n_g3 = (C_CH3_0 * ch3 + C_CH2_0 * ch2 + C_CH_0 * ch + C_C * c).* (1 .+ m_i.* sqrt.(T./ Tc))

        D    = (D_CH3_0 * V_ch3 + D_CH2_0 * V_ch2 + D_CH_0 * V_ch + D_C * V_c)

        #m_gc = sum(models[i].groups.n_groups[1])

        #D = D_i * m_gc

        y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)
        if any(!isfinite, y_pred)
            total += 1e20
            continue
        end

        # normalized SSE on eta = exp(y) - 1
        η      = exp.(y).- 1
        η_pred = exp.(y_pred).- 1
        total += sum(((η.- η_pred)./ η).^ 2) / length(y)
    end

    return isfinite(total) ? total : 1e20
end

# ------------------------------------------------------------
# Run CMA-ES to optimize CH parameters
# ------------------------------------------------------------
println("\nStarting CMA-ES optimization of CH parameters (A_CH, B_CH, C_CH, D_CH) on branched alkanes...")

σ0 = 1e-2
seed = 42
Random.seed!(seed)
stagnation_iters = 4000
iter_counter = Ref(0)

result = minimize(
    sse_C_only,
    x0,
    σ0;
    seed = seed,
    verbosity = 2,
    stagnation = stagnation_iters,
    maxiter = 10000,
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
final_sse  = fbest(result)
A_CH, B_CH, C_CH, D_CH = params_opt

println("\n✅ CMA-ES CH-only optimization successful!")
@printf "Optimized CH parameters:\n"
@printf "  A_CH = %.10f\n" A_CH
@printf "  B_CH = %.10f\n" B_CH
@printf "  C_CH = %.10f\n" C_CH
@printf "  D_CH = %.10f\n" D_CH
@printf "Final SSE = %.8e\n" final_sse
