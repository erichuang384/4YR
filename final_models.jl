function bell_lot_m_gc(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.10135275164382822, 0.01124217488372462, -0.06531812617372902, 0.00016441666190758258),
        "CH2" => (-1.1489661773542994, 0.011552481840714078, -0.007257467719755672,   0.000160555407606823),
        "CH"  => (   -5.76700392359268, 0.015484887480465368, 0.04629487021692841, 0.0016424307341824994),
		"C"  => (4.631164098326655, -0.0320619606463704, 0.14001914160206397, 0.00338275279484532)
    )
    γ =  0.17058657026926566
 
    Mw = Clapeyron.molecular_weight(model, z)

    num_groups = model.groups.n_groups[1]
    
    S = model.params.shapefactor

    m_gc_sf = sum(num_groups .* S)
    
 
    m0, m1, m2 = -1.4591110758580426, 0.6564832120302706, -0.021733593279943952

    m = m0 + m1 * m_gc_sf + m2 * m_gc_sf ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]

    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    crit_point = crit_pure(model)
    T_C = crit_point[1]
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*(T/T_C)^0.5)
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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

function bell_lot_m_gc_prop(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (-0.09191103851518649, 0.004927222521206559, -0.038858057532831364, 8.68028568322531e-5),
        "CH2" =>  (-0.8564259283122808, 0.00218618625031576, -0.005423577102369607, 0.0001506427406756405),
        "CH"  => (-1.0609438181149804, -0.008855568637209736, 0.029738051493388255, 0.001710762349323715),
		"C"  => (   9.462410337187325, -0.03347424902347585, 0.08102340863545922, 0.004081780859332927))
    
    γ =  -0.04953052132243503
 
    Mw = Clapeyron.molecular_weight(model, z)

    num_groups = model.groups.n_groups[1]
    
    S = model.params.shapefactor

    m_gc_sf = sum(num_groups .* S)
    
 
    m0, m1, m2 = -1.6137485955606357, 0.7643529685448124, -0.01219637283336471

    m = m0 + m1 * m_gc_sf + m2 * m_gc_sf ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]

    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    crit_point = crit_pure(model)
    T_C = crit_point[1]
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*(T/T_C)^0.5)
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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

function bell_lot_test(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.14544200778907007, 0.010683439643496976, -0.06148080798139255, 0.000157392327889979),
        "CH2" => (-1.1927784860733681, 0.011458247757004004, -0.008458302429596044,  0.00016724013588380074),
        "CH"  => (    -6.70322582565162, 0.013433432693828238, 0.03532440730533404, 0.0016759054663535538),
		"C"  => (   5.441006090748911, -0.0513546650750801, 0.1412593290192207, 0.0026320850844137205)
    )
    γ = 0.16499191915814165
 
    Mw = Clapeyron.molecular_weight(model, z)
 
    m0, m1, m2 = -1.019572741285313, 10.667125465466807, -7.956775819220652

    m = m0 + m1 * Mw + m2 * Mw ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]
    num_groups = model.groups.n_groups[1]
    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    crit_point = crit_pure(model)
    T_C = crit_point[1]
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*sqrt(T/T_C))
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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

function bell_lot_redcrit(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.12824947582505702, 0.013076076373454747, -0.04978810903761773, 0.00013127464119180518),
        "CH2" => ( -1.1790646754884186, 0.01530877542575063, -0.008718576449895543,  0.0001592474995354457),
        "CH"  => ( -6.430291253854435, 0.03864215679822269, 0.045835943423648294, 0.0001592474995354457),
		"C"  => (4.258766667381834, -0.019423064027923164, 0.10686284556181676, 0.004399162994060731)
    )
    γ = 0.21489488076369473
 
    Mw = Clapeyron.molecular_weight(model, z)

    crit_point = crit_pure(model)
    T_C = crit_point[1]

    reduced_crit_temp = T_C / ϵ_OFE(model)
 
    m_0, m_1, m_2 =  -1.0004327993538618, -1.5250909780948594, 1.4351059929796646

    m = m_0 + m_1 * reduced_crit_temp + m_2 * reduced_crit_temp ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]
    num_groups = model.groups.n_groups[1]
    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*(T/T_C)^0.5)
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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

function bell_lot_exp1_MW(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.2174584700616018, 0.008361680913211746, -0.025053309530719003, 0.00011170753171621334),
        "CH2" => ( -1.1949802524204984, 0.009496373781074274, -0.01887191332418902,  0.00017118748215993833),
        "CH"  => (   -8.174665043147424, 0.020268968800656033, -0.037029453668895734, 0.001993761553613756),
		"C"  => ( 2.9338462417947166, -0.052685137143649134, 0.08362576947791169, 0.0021440131362818115)
    )
    γ = 0.13710940660288795
 
    Mw = Clapeyron.molecular_weight(model, z)
 
    m0, m1, m2 = -0.6190192697993737, 5.77318771388499, -4.20985789676712

    m = m0 + m1 * Mw + m2 * Mw ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]
    num_groups = model.groups.n_groups[1]
    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    crit_point = crit_pure(model)
    T_C = crit_point[1]
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*(T/T_C))
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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

function bell_lot_exp1_m_gc(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.22541973654155878, 0.0003573272430184375, 0.000695199527103719, 6.786493990364583e-5),
        "CH2" => ( -0.7356974157625014, 5.372137601949603e-5, 1.5877694824987887e-5,   5.066876297097107e-5),
        "CH"  => (  -3.1321656310571955, -0.001177153095142731, -0.0007073080172634057, 0.0011741851891547293),
		"C"  => (1.00495352932324, -0.003391865190510336, -0.001854009111603349, 0.003112169830586356)
    )
    γ =  -0.5216114752368064
 
    Mw = Clapeyron.molecular_weight(model, z)

    num_groups = model.groups.n_groups[1]
    m_gc = sum(num_groups)
    
 
    m0, m1, m2 =  -11.588829753617839, 9.61413681409126, -0.8500287293823466

    m = m0 + m1 * m_gc + m2 * m_gc ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]

    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    crit_point = crit_pure(model)
    T_C = crit_point[1]
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*(T/T_C))
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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



function bell_lot_exp1_acen(model::EoSModel, P, T, z = StaticArrays.SA[1.0]; params = nothing)
    n_alpha = Dict(
        "CH3" => (0.27599553365709784, 0.006709417053223589, -0.009178326922009292, 6.113531926494879e-5),
        "CH2" => (-1.1896522318659597, 0.009390359501030357, -0.02097893557821805,   0.0001828343650515877),
        "CH"  => ( -9.081508102900115, 0.02855906756928941, -0.06784500497619946, 0.002359342904642352),
		"C"  => (1.2124070698351408, -0.031242082328243003, 0.04332245635190416, 0.002858888200800245)
    )
    γ =  0.13135512942533029
 
    Mw = Clapeyron.molecular_weight(model, z)

    num_groups = model.groups.n_groups[1]
    
    S = model.params.shapefactor

    m_gc_sf = sum(num_groups .* S)
    acentric_fact = acentric_factor(model)
    
 
    m0, m1, m2 =  -0.7284465485940593, 2.3272599727351, -0.43721020395924043

    m = m0 + m1 * acentric_fact + m2 * acentric_fact ^ 2

    A_i = Dict(k => v[1] for (k,v) in n_alpha)
 
    groups = model.groups.groups[1]

    S = model.params.shapefactor
    σ = diag(model.params.sigma.values) .* 1e10
 
 
    n_g_matrix = zeros(length(groups), 4)
    V = sum(num_groups .* S .* (σ .^ 3))
 
    crit_point = crit_pure(model)
    T_C = crit_point[1]
 
    for i in 1:length(groups)
        gname = groups[i]
        if !haskey(n_alpha, gname)
            error("Group $gname not found in parameter dictionary")
        end
        A, B, C, D = n_alpha[gname]
 
        n_g_matrix[i, 1] = 0
        n_g_matrix[i, 2] = B * S[i] * σ[i]^3 * num_groups[i] / (V^γ)
        n_g_matrix[i, 3] = C * num_groups[i] * (1 + m*(T/T_C))
        n_g_matrix[i, 4] = D * S[i] * σ[i]^3 * num_groups[i]
    end
 
    n_g = vec(sum(n_g_matrix, dims = 1))
    n_g[1] = A_vdw(model, A_i)
 
    tot_sf = sum(S .* num_groups)
 
    R = Clapeyron.Rgas()
 
    s_res = entropy_res(model, P, T, z)
 
    Z = (-s_res) / (R * tot_sf)
 
    ln_n_reduced = n_g[1] + n_g[2] * Z + n_g[3] * Z^2 + n_g[4] * Z^3
 
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