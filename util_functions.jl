using LinearAlgebra, StaticArrays
function entropy_ideal(model::EoSModel, p , T , z = StaticArrays.SA[1.0])
    s_id = Clapeyron.entropy(model, p, T, z) - entropy_res(model, p, T, z)
    return s_id
end

function global_crit_dict(models)
    crit_dict = Dict{String, Tuple{Float64, Float64}}()

    for i in 1:length(models)
        crit_point = crit_pure(models[i])  # returns (P_c, T_c)

        model_name = models[i].components[1]  # Use component name as key
        crit_dict[model_name] = (crit_point[2], crit_point[1])
    end
    global crit_dict
    return 
end

function entropy_res_crit(model::EoSModel)
    """
    Compute residual entropy at the critical point,
    using precomputed critical T and P from CRIT_DICT.
    """
    model_name = string(model.components[1])

    if haskey(crit_dict, model_name)
        P_c, T_c = crit_dict[model_name]
    end

    res_crit_entr = entropy_res(model, P_c, T_c)
    return res_crit_entr
end

function csv_phase(csv_path::String, model, remove_vapour::Bool=false)
    # Read CSV
    df = CSV.read(csv_path, DataFrame)
    
    # Extract pressure and temperature columns (1st and 2nd)
    pressure = df[:, 1]
    temperature = df[:, 2]
    
    # Identify phase using Clapeyron
    phases = Clapeyron.identify_phase.(model, pressure, temperature)
    
    # Add phase column
    df[!, :Phase] = phases
    
    # Optionally remove vapour phase rows
    if remove_vapour
        df = filter(row -> row.Phase != :vapour, df)
    end
    vapour_count = count(==( :vapour ), phases)
    println("Number of vapour phase entries: $vapour_count")
    # Write back to the same CSV
    CSV.write(csv_path, df)

    return df
end


function σ_OFE(model::EoSModel)
    """
    sigma pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    σ = cbrt(xₛₖ' * (model.params.sigma).^3 * xₛₖ)
    return σ
end

function ϵ_OFE(model::EoSModel)
    """
    epsilon pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    ϵ_kl = model.params.epsilon.values
    ϵ = xₛₖ' * ϵ_kl * xₛₖ
    return ϵ
end

function λ_r_OFE(model::EoSModel)
    """
    λ_r pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    λ_kl = model.params.lambda_r.values
    λ = xₛₖ' * λ_kl * xₛₖ
    return λ
end

function  a²²ᵢ(λ_r)
    """
    Matrix of coefficients for Collision integral by Fokin et al.
    """

    a²² = [0.0           0.113086e1    0.234799e2    0.310127e1;
    0.0           0.551559e1   -0.137023e3    0.185848e2;
    0.325909e-1  -0.292925e2    0.243741e3    0.0;
    0.697682      0.590192e2   -0.143670e3   -0.123518e3;
   -0.564238     -0.430549e2    0.0           0.137282e3;
    0.126508      0.104273e2    0.150601e2   -0.408911e2]
    a = zeros(6)
    for i in 1:6
        a[i] = a²²[i,1] + a²²[i,2]/λ_r + a²²[i,3]/λ_r^2 + a²²[i,4]/λ_r^3
    end
    return a
end

function Ω⃰_LJ(model::EoSModel,T)
    """
    Reduced Collision Integral
    For two component, need to work on getting epsilon in general
    """
    
    ϵ = ϵ_OFE(model)
    T⃰ = T / ϵ
    if !(0.3 < T⃰ < 100)
        print("Unviable T for correlation")
    end
    #think that k_B is already in epsilon parameter according to sample calc
    Ω = 1.16145*(T⃰)^(-0.14874)+0.52487*exp(-0.77320*(T⃰))+2.16178*exp(-2.43787*(T⃰))
    return Ω
end

function x_sk(model::EoSModel)
    """
    x_sk
    """
    s = model.params.shapefactor
    segment = model.params.segment
    gc = 1:length(model.groups.flattenedgroups)
    comps = 1:length(model.groups.components)
    v = float.(model.groups.n_groups)
    for k in 1:length(gc)
        for i in 1:length(comps)
            v[i][k] = v[i][k]*s[k]*segment[k]
        end
    end
    xₛₖ = v./sum(v[1])
    return xₛₖ[1]
end

function x_sk_mix(model::EoSModel, z)
    # mixture group fractions x_{s,k} from mole fractions z (length = #components)
    components = model.components

    for g in 1:length(components)
        comp_model = SAFTgammaMie([components[g]])
        S = comp_model.params.shapefactor
        segment = comp_model.params.segment
    end


    s = model.params.shapefactor
    segment = model.params.segment
    gc = 1:length(model.groups.flattenedgroups)     # group indices (k = 1..N_G)
    comps = 1:length(model.groups.components)       # component indices (i = 1..N_C)

    # copy group counts ν_{k,i} and weight by ν_k^* S_k
    v = float.(model.groups.n_groups)               # v[i][k] = ν_{k,i}
    for k in 1:length(z)
        for i in comps
            v[i][k] = v[i][k] * segment[k] * s[k]   # ν_{k,i} ν_k^* S_k
        end
    end

    # numerator per group: Σ_i x_i ν_{k,i} ν_k^* S_k
    numer = zeros(Float64, length(gc))
    for k in 1:length(z)
        for i in comps
            numer[k] += z[i] * v[i][k]
        end
    end

    # denominator: Σ_i x_i Σ_l ν_{l,i} ν_l^* S_l  = sum(numer)
    denom = sum(numer)
    denom = denom == 0.0 ? eps(Float64) : denom

    xs = numer./ denom
    return xs
end

function x_sk_mix(model::EoSModel, z)
    # Mixture group fractions x_{s,k} from mole fractions z
    # Returns vector in the order of model.groups.flattenedgroups

    # Components and global group order
    components = model.groups.components
    g_global   = model.groups.flattenedgroups
    ng         = length(g_global)
    nc         = length(components)
    @assert length(z) == nc "Length of z ($(length(z))) must match number of components ($nc)."

    # Map group name -> global index
    gidx = Dict(g_global[k] => k for k in 1:ng)

    # V[i,k] = ν_{k,i} * ν_k^* * S_k (mapped to global group index k)
    V = zeros(Float64, nc, ng)

    for i in 1:nc
        comp_model = SAFTgammaMie([components[i]])
        groups     = comp_model.groups.groups[1]          # local group names for component i
        n_grp      = comp_model.groups.n_groups[1]        # local counts ν_{k,i}
        s          = comp_model.params.shapefactor        # local S_k
        segment    = comp_model.params.segment            # local ν_k^*

        for j in 1:length(groups)
            gname = groups[j]
            k     = gidx[gname]                           # map to global group index
            V[i,k] += float(n_grp[j]) * segment[j] * s[j] # ν_{k,i} ν_k^* S_k
        end
    end

    # Numerator per group: Σ_i x_i V[i,k]
    numer = zeros(Float64, ng)
    for k in 1:ng
        for i in 1:nc
            numer[k] += z[i] * V[i,k]
        end
    end

    # Denominator: Σ_i x_i Σ_l V[i,l] = sum(numer)
    denom = sum(numer)
    denom = denom == 0.0 ? eps(Float64) : denom

    return numer./ denom
end

function Ω⃰(model::EoSModel, T)
    """
    Collision Integral correlation by Fokin et al.
    """
    λ_r = λ_r_OFE(model)
    a_vals =  a²²ᵢ(λ_r)
    ϵ = ϵ_OFE(model)
    T⃰ = T / ϵ

    ln_Omega = -2/λ_r * log(T⃰) + log(1 - 2/(3*λ_r)) + sum(a_vals[i] * (1/T⃰)^((i-1)/2) for i in 1:6)

    return exp(ln_Omega)
end

function tau_OFE(model::EoSModel, tau_i)
    """
    epsilon pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    x_sk_order = model.groups.flattenedgroups
    x_sk_dict = Dict(x_sk_order[i] => xₛₖ[i] for i in eachindex(x_sk_order))
    #x_sk_vec   = [x_sk_dict[g] for g in x_sk_order] # ensure in same order
    tau_vec = [tau_i[g] for g in x_sk_order] # has to be matrix

    n = length(x_sk_order)
    tau_mat = zeros(n,n)
    for i in 1:n, j in 1:n
        tau_mat[i, j] = (i == j) ? tau_vec[i] : sqrt(tau_vec[i] * tau_vec[j]) # using mean value
    end

    x_vec = model.groups.n_groups[1]./sum(model.groups.n_groups[1])
    #vdW mixing rule
    #tau_ofe = (x_sk_vec' * (tau_mat.^1) * x_sk_vec)
    tau_ofe = sum(tau_vec .* x_vec)
    return tau_ofe
end

function xi_OFE(model::EoSModel, ξ_i)
    """
    epsilon pure fluid equivalent
    """
    xₛₖ = x_sk(model)
    x_sk_order = model.groups.flattenedgroups
    x_sk_dict = Dict(x_sk_order[i] => xₛₖ[i] for i in eachindex(x_sk_order))
    x_sk_vec   = [x_sk_dict[g] for g in x_sk_order] # ensure in same order
    xi_vec = [ξ_i[g] for g in x_sk_order] # has to be matrix

    n = length(x_sk_order)
    xi_mat = zeros(n,n)
    for i in 1:n, j in 1:n
        xi_mat[i, j] = (i == j) ? xi_vec[i] : sqrt(xi_vec[i] * xi_vec[j]) # using mean value
    end

    #x_vec = model.groups.n_groups[1]./sum(model.groups.n_groups[1])
    #vdW mixing rule
    tau_ofe = (x_sk_vec' * (xi_mat.^1) * x_sk_vec)
    #tau_ofe = sum(tau_vec .* x_vec)
    return tau_ofe
end

function load_experimental_data(path::AbstractString)
    """
    Format experimental data from CSV
    """
    df = CSV.read(path, DataFrame)
    # normalize column names to lowercase symbols
    rename!(df, Symbol.(lowercase.(String.(names(df)))))
    required = [:p, :t, :viscosity]
    #for c in required
    #    @assert c ∈ names(df) "Missing column: $c in $path. Expected columns: $(required)."
    #end
    return df
end

function A_vdw_opt(model::EoSModel, A_CH3, A_CH2,A_CH, A_C)
    """
    epsilon pure fluid equivalent
    """
    A_i = Dict("CH3" => A_CH3, "CH2" => A_CH2,"CH" =>A_CH, "C" =>A_C)
    #x_sk_order = ["CH3", "CH2",]
    xₛₖ = x_sk(model)
    x_sk_order = model.groups.flattenedgroups
    x_sk_dict = Dict(x_sk_order[i] => xₛₖ[i] for i in eachindex(x_sk_order))
    x_sk_vec   = [x_sk_dict[g] for g in x_sk_order] # ensure in same order
    A_vec = [A_i[g] for g in x_sk_order] # has to be matrix

    n = length(x_sk_order)
    A_mat = zeros(n,n)
    for i in 1:n, j in 1:n
        A_mat[i, j] = (i == j) ? A_vec[i] : (A_vec[i] + A_vec[j])/2 # using mean value
    end

    #x_vec = model.groups.n_groups[1]./sum(model.groups.n_groups[1])
    #vdW mixing rule
    
    A_ofe = (x_sk_vec' * (A_mat.^1) * x_sk_vec)
    #tau_ofe = sum(tau_vec .* x_vec)
    return A_ofe
end

function A_vdw(model::EoSModel, A_i)
    """
    epsilon pure fluid equivalent
    """
    
    xₛₖ = x_sk(model)
    x_sk_order = model.groups.flattenedgroups
    x_sk_dict = Dict(x_sk_order[i] => xₛₖ[i] for i in eachindex(x_sk_order))
    x_sk_vec   = [x_sk_dict[g] for g in x_sk_order] # ensure in same order
    A_vec = [A_i[g] for g in x_sk_order] # has to be matrix

    n = length(x_sk_order)
    A_mat = zeros(n,n)
    for i in 1:n, j in 1:n
        A_mat[i, j] = (i == j) ? A_vec[i] : (A_vec[i] + A_vec[j])/2 # using mean value
    end

    #x_vec = model.groups.n_groups[1]./sum(model.groups.n_groups[1])
    #vdW mixing rule
    
    A_ofe = (x_sk_vec' * (A_mat.^1) * x_sk_vec)
    #tau_ofe = sum(tau_vec .* x_vec)
    return A_ofe
end

function A_vdw_mix(model::EoSModel, A_i,z)
    """
    epsilon pure fluid equivalent
    """
    
    xₛₖ = x_sk_mix(model,z)
    n = length(z)
    A_vdw_pure = zeros(n)
    components = model.components
    for i in 1:n
        comp_model = SAFTgammaMie([components[i]])
        A_vdw_pure[i] = A_vdw(comp_model, A_i)
    end

    A_mat = zeros(n,n)

    for i in 1:n, j in 1:n
        A_mat[i, j] = (i == j) ? A_vdw_pure[i] : (A_vdw_pure[i] + A_vdw_pure[j])/2 # using mean value
    end

    #x_vec = model.groups.n_groups[1]./sum(model.groups.n_groups[1])
    #vdW mixing rule
    
    A_ofe_mix = (z' * (A_mat.^1) * z)
    #tau_ofe = sum(tau_vec .* x_vec)
    return A_ofe_mix
end