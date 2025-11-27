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
   # SAFTgammaMie(["Ethane"]),
   SAFTgammaMie(["Propane"]),
    SAFTgammaMie(["Butane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Pentane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Octane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Decane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Dodecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Tridecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Pentadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Hexadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Heptadecane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["Eicosane"], idealmodel = BasicIdeal),
    SAFTgammaMie(["2,6,10,14-tetramethylpentadecane"]),
    SAFTgammaMie(["2-methylpropane"]),
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
    SAFTgammaMie(["7-methylhexadecane"]),
    SAFTgammaMie(["2,2,4-trimethylpentane"]),
    SAFTgammaMie(["heptamethylnonane"]),
    SAFTgammaMie(["2,2,4-trimethylhexane"]),
    SAFTgammaMie(["Neopentane"]), # 2,2-dimethylpropane
    SAFTgammaMie(["2,2-dimethylbutane"]),
    SAFTgammaMie(["2-methylpropane"]),
    SAFTgammaMie(["3-methylpentane"]),
    SAFTgammaMie(["2,3-dimethylbutane"]),
    SAFTgammaMie(["3-ethylpentane"]),
    SAFTgammaMie(["2,4-dimethylpentane"]),
    SAFTgammaMie(["2,3,4-trimethylpentane"])
]


data_files = [
   # "Training DATA/ethane.csv",
   "Training DATA/Propane DETHERM.csv",
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
    "Training DATA/n-eicosane.csv",
    "Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv",
   "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/2-methylbutane.csv",
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
    "Training DATA/Branched Alkane/7-methylhexadecane.csv",
    "Training DATA/Branched Alkane/2,2,4-trimethylpentane.csv",
    "Training DATA/Branched Alkane/2,2,4,4,6,8,8-heptamethylnonane.csv",
    "Training DATA/Branched Alkane/2,2,4-trimethylhexane.csv",
    "Training DATA/Branched Alkane/2,2-dimethylpropane.csv",
    "Training DATA/Branched Alkane/2,2-dimethylbutane.csv",
    "Training DATA/Branched Alkane/2-methylpropane.csv",
    "Training DATA/Branched Alkane/3-methylpentane.csv",
    "Training DATA/Branched Alkane/2,3-dimethylbutane.csv",
    "Training DATA/Branched Alkane/3-ethylpentane.csv",
    "Training DATA/Branched Alkane/2,4-dimethylpentane.csv",
    "Training DATA/Branched Alkane/2,3,4-trimethylpentane.csv"
]
weightings = [0.8, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0, 0.8,
0.15, 0.35, 0.45, 0.55, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 , 0.3 , 0.3, 0.3, 0.4, 0.1]

weightings = [0.4, 0.6, 0.9, 1.0, 1.0, 1.0, 0.9,1.0,1.0,1.0,1.0, 0.8,
0.2, 0.35, 0.45, 0.55, 0.55, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 , 0.3 , 0.3, 0.3, 0.4, 0.1, 0.1, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1]
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

# for using mgc in tau
#x0 = [0.3331146804411593, 0.004008516939816172, -7.200729416054543e-5, -0.8790852726537773, 0.007301853141179978, 3.548541125552805e-5, 
#-6.54349752948493, 0.02691391753786118, 0.00020523872873219124, -2.672989576306306, 5.156256391072562e-6, 9.079965247259052e-5, 0.13414255828609212, 1.369922923059573e-5, 0.000101750454318113, 0.0017891590557210708, 0.003178772122537697, -0.0, -27.23026925611101, 0.3356077557831271]

# for Mw m_i
#x0 = [0.22094535978490432, 0.008333968719790204, -0.024182974132460986, -1.1978126548498873, 0.00958546475384391, -0.019137708023955466, -8.236926954802959, 0.021009247506700666, -0.03887237012467386, 2.770012802937354, -0.052121424309366504, 0.08202924893856159, 0.13816670401137154, 0.00010971206060431459, 0.00017195443344511583, 0.0020095458507893896, 0.0021533341333265558, -0.6123816709188983, 5.703776593787865, -4.148366576410572]
# for acentric_fact m_i
#x0 =   [0.18380429041626706, 0.00926271817583487, -0.016736791143933925, -1.1628779418673048, 0.00971090571491477, -0.013748613245391289, -7.74673863485519, 0.016728450066019186, -0.029417127854903555, 5.201552330552332, -0.06115356102481718, 0.05215664709797355, 0.142522263832269, 5.213959989501366e-5, 0.00018572431352956148, 0.0022188349152456053, 0.003361644345102922, -0.43693000399985965, 3.093202568036602, 0.0]

# for res_crit -1 
#x0 =  [0.8323655477410299, -0.058045759242486276, 0.052514146734784724, -1.640918108637263, 0.11326417950757697, -0.02962635224076876, -21.290840211647932, 1.1865667944789025, -0.18663604824732855, -37.659105971236144, 2.1396093794863424, -0.26359791797455423, 0.4782629614789716, -0.0001840795444819355, 0.0002656538948040817, 0.004418155234869952, 0.01169573616754786, 0.0, 0.0, 0.0]
# for m_gc
#x0 = [0.22541973654155878, 0.0003573272430184375, 0.000695199527103719, -0.7356974157625014, 5.372137601949603e-5, 1.5877694824987887e-5, -3.1321656310571955, -0.001177153095142731, -0.0007073080172634057, 1.00495352932324, -0.003391865190510336, -0.001854009111603349, -0.5216114752368064, 6.786493990364583e-5, 5.066876297097107e-5, 0.0011741851891547293, 0.003112169830586356, -132.6016447753526, 73.8202785509046, -6.205828665432313]
# for m_gc_ sf
#x0 = [0.1793909059682201, 0.009906024147471784, -0.032443678340918304, -1.1954468646686496, 0.010866200048007328, -0.018069176391011424, -7.433113193147548, 0.020282876531754115, -0.018754171250769915, 4.676828161429562, -0.05952227796255291, 0.09285522064040783, 0.15955580768051603, 0.00013442194288733526, 0.00016692356987303716, 0.0017789422686279874, 0.0022823298212525903, -0.8188231498626467, 0.34853650297571487, -0.013065704734308384]
x0 = [0.10135275164382822, 0.01124217488372462, -0.06531812617372902, -1.1489661773542994, 0.011552481840714078, -0.007257467719755672, -5.76700392359268, 0.015484887480465368, 0.04629487021692841, 4.631164098326655, -0.0320619606463704, 0.14001914160206397, 0.17058657026926566, 0.00016441666190758258, 0.000160555407606823, 0.0016424307341824994, 0.00338275279484532, -1.4591110758580426, 0.6564832120302706, -0.021733593279943952]
x0 = [0.10135275164382822, 0.01124217488372462, -0.06531812617372902, 0.00016441666190758258, -1.1489661773542994, 0.011552481840714078, -0.007257467719755672, 0.000160555407606823,-5.76700392359268, 0.015484887480465368, 0.04629487021692841, 0.0016424307341824994, 4.631164098326655, -0.0320619606463704, 0.14001914160206397, 0.00338275279484532, 0.17058657026926566,    -1.4591110758580426, 0.6564832120302706, -0.021733593279943952]

#x0 =  [0.276172545733216, 0.004323441060546311, 0.0051360218264248383, -0.858139784605485, 0.004971039401045132, 0.0018174656880325422, -5.059531579711798, 0.01218190481581754, -0.004908863798711147, 2.09587776942617, -0.0054535617130707, -0.01127564552081357, 0.08374175695290709, 6.8552731116276995e-6, 8.57862254931154e-5, 0.001821910924104962, 0.004892380139053853, -0.091644661882207, 12.69945876312193, -8.908843912993759]
# Optimize with ethane
#x0 = [-0.6739365301469833, 0.013456670788197329, -0.11188727933703324, -0.707694791261234, 0.003263574186063659, 0.0008704094064981151, 
#4.292645876423649, -0.038217663156377375, 0.11270181557358286, 32.15734270198932, -0.13660198799657275, 0.27210026116495456, 0.044099231101498014, 0.00028428080597803326, 0.00013375467681705678, 0.0008521909560508448, 0.0009537809314066211, -0.5060292649221828, 3.7978209930108053, 14.168138509769243]
# For exponent
#x0 =  [0.34531907584036703, 0.00589259764590633, 0.005449876374023697, -1.2495317600879672, 0.011325029954147218, -0.025785723899260555, -10.450313188259628, 0.047265679011482924, -0.10450975047560372, -2.468383250164794, -0.014099829015488495, 0.016092510067068758, 0.15347012249968817, 2.9441047082159807e-5, 0.00019470190318104943, 0.002686320906500846, 0.002971626547105782, -0.6186913405953289, 1.719106761463683, -0.02181427729655787, 1.4109206041079203]





#= 
x0_base =  [0.27599553365709784, 0.006709417053223589, -0.009178326922009292, -1.1896522318659597, 0.009390359501030357, -0.02097893557821805, -9.081508102900115, 0.02855906756928941, -0.06784500497619946, 1.2124070698351408, -0.031242082328243003, 0.04332245635190416, 0.13135512942533029, 6.113531926494879e-5, 0.0001828343650515877, 0.002359342904642352, 0.002858888200800245, 1.0, -0.7284465485940593, 2.3272599727351, -0.43721020395924043]

x0 = vcat(x0_base, zeros(length(models))) =#

println("Initial CH guess:")
#@printf "  A_CH = %.10f, B_CH = %.10f, C_CH = %.10f, D_CH = %.10f\n" x0...

# ------------------------------------------------------------
# Objective: optimize CH params only (A_CH, B_CH, C_CH, D_CH)
# CH shapefactor S_ch is taken as model constant (S_ch_model_list), not optimized
# Additive contributions from CH3 + CH2 + CH for n_g2, n_g3, D
# ------------------------------------------------------------
#= function sse_all(params::AbstractVector{<:Real})
    NM = length(models)
    if length(params) != 21 + NM
        return 1e20
    end

    # unpack first 20 parameters (same as before):
    A_CH3, B_CH3, C_CH3,
    A_CH2, B_CH2, C_CH2,
    A_CH,  B_CH,  C_CH,
    A_C,   B_C,   C_C,
    gamma_i,
    D_CH3, D_CH2, D_CH, D_C, p,
    dummy1, dummy2, dummy3 = params[1:21]   # unused old m1,m2,m3

    total = 0.0

    for i in 1:NM
        z = data_z_list[i]
        y = data_y_list[i]
        T = data_T_list[i]
        if isempty(z); continue; end

        # Get model-specific m_i directly from params:
        m_i = params[21 + i]

        # Precomputed group structure:
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
        xsk = x_sk_list[i]
        Mw  = Mw_list[i]
        m_gc = m_gc_list[i]
        acentric_fact = acentric_factor_list[i]
        res_entr_crit = res_crit_entr_list[i]

        # Volumes
        V_ch3 = ch3 * S_ch3 * sigma_ch3
        V_ch2 = ch2 * S_ch2 * sigma_ch2
        V_ch  = ch  * S_ch * sigma_ch
        V_c   = c   * S_c * sigma_c

        V_tot = V_ch3 + V_ch2 + V_ch + V_c
        if !(isfinite(V_tot)) || V_tot <= 0
            total += 1e20
            continue
        end

        n_g1 = A_vdw_opt(models[i], A_CH3, A_CH2, A_CH, A_C)

        n_g2 = ((B_CH3 * V_ch3) + (B_CH2 * V_ch2) + (B_CH * V_ch) + (B_C * V_c)) / V_tot^gamma_i

        # *** NEW: directly use per-model m_i ***
        n_g3 = (C_CH3*ch3 + C_CH2*ch2 + C_CH*ch + C_C*c) .* (1 .+ m_i .* (T ./ Tc) .^ p)

        D = (D_CH3 * V_ch3 + D_CH2 * V_ch2 + D_CH * V_ch + D_C * V_c)

        y_pred = n_g1 .+ n_g2 .* z .+ n_g3 .* (z.^2) .+ D .* (z.^3)

        η      = exp.(y) .- 1.0
        η_pred = exp.(y_pred) .- 1.0

        total += (sum(((η .- η_pred) ./ η).^2) / length(y)) * weightings[i]
    end

    return isfinite(total) ? total : 1e20
end =#

function sse_all(params::AbstractVector{<:Real})
    if length(params) != 20
        return 1e20
    end
    #A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, A_CH, B_CH, C_CH, gamma_i, D_i, m_1, m_2, m_3 = params
    A_CH3, B_CH3, C_CH3, D_CH3, A_CH2, B_CH2, C_CH2, D_CH2, A_CH, B_CH, C_CH, D_CH, A_C, B_C, C_C, D_C, gamma_i, m_1, m_2, m_3 = params
    #A_CH3, B_CH3, C_CH3, A_CH2, B_CH2, C_CH2, A_CH, B_CH, C_CH, A_C, B_C, C_C, gamma_i, D_CH3, D_CH2, D_CH, D_C, m_1, m_2, m_3 = params
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
        #xsk = x_sk_list[i]
        Mw  = Mw_list[i]
        m_gc = m_gc_list[i]
        #acentric_fact = acentric_factor_list[i]
        #res_entr_crit = res_crit_entr_list[i]
        red_crit_temp = reduced_crit_temp[i]

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
        #n_g1 = A_vdw_opt(models[i], A_CH3, A_CH2,A_CH,A_C)
        n_g1       = (A_CH3 * V_ch3) + (A_CH2 * V_ch2) + (A_CH * V_ch) + (A_C * V_c)

        n_g2= ((B_CH3 * V_ch3) + (B_CH2 * V_ch2) + (B_CH * V_ch) + (B_C * V_c))/V_tot^gamma_i
        #n_g2     = n_g2_num / V_tot^gamma_fix


       # m_i  = m_1 + m_2*acentric_fact + m_3*acentric_fact^2

        #m_i  = m_1 + m_2*(res_entr_crit) + m_3*(res_entr_crit)^2  

        #m_i  = m_1 + m_2*Mw + m_3*Mw^2

        m_i  = m_1 + m_2*m_gc + m_3*m_gc^2
        #m_i  = m_1 + m_2*red_crit_temp + m_3*red_crit_temp^2

        n_g3 = (C_CH3 * ch3 + C_CH2 * ch2 + C_CH * ch + C_C * c).* (1 .+ m_i.* (T./ Tc) .^ 0.5) 

        D    = (D_CH3 * V_ch3 + D_CH2 * V_ch2 + D_CH * V_ch + D_C * V_c)

        #m_gc = sum(models[i].groups.n_groups[1])

        #D = D_i * m_gc

        y_pred = n_g1.+ n_g2.* z.+ n_g3.* (z.^ 2).+ D.* (z.^ 3)
        if any(!isfinite, exp.(y_pred))
            total += 1e10
            continue
        end

        # normalized SSE on eta = exp(y) - 1
        η      = exp.(y) .- 1.0
        η_pred = exp.(y_pred) .- 1.0
        total += (sum(((η.- η_pred)./ η).^ 2) / length(y) ) * weightings[i]
    end

    return isfinite(total) ? total : 1e20
end

# ------------------------------------------------------------
# Run CMA-ES to optimize CH parameters
# ------------------------------------------------------------
println("\nStarting CMA-ES optimization of CH parameters (A_CH, B_CH, C_CH, D_CH) on branched alkanes...")

σ0 = 0.1
seed = 142
Random.seed!(seed)
stagnation_iters = 100000
iter_counter = Ref(0)

lower = fill(-Inf, 20)
upper = fill(Inf, 20)
lower[18:20] .= -15.0
upper[18:20] .= 15.0

result = minimize(
    sse_all,
    x0,
    σ0;
    seed = seed,
    verbosity = 2,
    stagnation = stagnation_iters,
    maxiter = 50000,
    ftol = 1e-15,
    lower = lower,
    upper = upper,
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