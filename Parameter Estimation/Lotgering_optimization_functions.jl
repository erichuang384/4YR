include("util_functions.jl")

function Lotgering_viscosity(model::EoSModel, P, T)
    """
    Lotgering method for pure fluid
    (combines reduced viscosity and dilute gas contribution)
    """

    # --- Extract group and parameter data ---
    n_α = model.groups.n_groups[1]
    S = model.params.shapefactor
    #S = [0.644362, 0.384329]
    σ = diag(model.params.sigma.values) .* 1e10

    # --- Lotgering coefficients ---
    #x = [-0.001555382, -0.244068, -0.00589621]
    a_α = [-0.00617, -0.00156]
    b_α = [-0.19444, -0.24407]
    c_α = [-0.0597, -0.0059]
    #d_α = [-0.01245, -0.01245]
    γ = 0.45
    Mw = Clapeyron.molecular_weight(model)

    # --- Intermediate terms ---
    V = sum(n_α .* S .* (σ .^ 3))
    A = sum(n_α .* S .* (σ .^ 3) .* a_α)
    B = sum((n_α .* S .* (σ .^ 3) .* b_α)) / (V ^ γ)
    C = sum(n_α .* c_α)
    D = (-1.25594-888.1232/Mw) ^ (-1) 

    # --- Entropy and reduced viscosity ---
    s_res = entropy_res(model, P, T)
    m_gc = sum(model.groups.n_groups[1])
    R = Rgas()
    z = s_res / (R * m_gc)  # molar entropy term

    viscosity_reduced = exp(A + B * z + C * z^2 + D * z^3)

    # --- Combine with dilute gas viscosity ---
    viscosity_CE = IB_CE(model, T)
    viscosity = viscosity_reduced * viscosity_CE

    return viscosity
end


include("util_functions.jl")
#=
function Lotgering_viscosity_optimize(model::EoSModel, P, T; params = Dict(:a_α => [-8.6878e-3, -0.9194e-3],
                                                                              :b_α => [-1.7951e-1, -1.3316e-1],
                                                                              :c_α => [-12.2359e-2, -4.2657e-2]))
    """
    Lotgering method for pure fluid (combined).
    Expects params Dict with symbol keys :a_α, :b_α, :c_α mapping to 2-element vectors.
    """

    # --- Extract group and parameter data ---
    n_α = model.groups.n_groups[1]
    # If you want to use model.params.shapefactor instead, revert this line
     S = model.params.shapefactor

    # ensure sigma is a vector (not a Diagonal matrix)
    σ = diag(model.params.sigma.values) .* 1e10

    # --- Lotgering coefficients from params dict (each should be a 2-element vector) ---
    a_α = params[:a_α]
    b_α = params[:b_α]
    c_α = params[:c_α]

    # fixed values
    d_α = [-0.01245, -0.01245]
    γ = 0.45
    Mw = Clapeyron.molecular_weight(model)
    # --- Intermediate terms (elementwise) ---
    V = sum(n_α .* S .* (σ .^ 3))
    A = sum(n_α .* S .* (σ .^ 3) .* a_α)
    B = sum(n_α .* S .* (σ .^ 3) .* b_α) / (V ^ γ)
    C = sum(n_α .* c_α)
    D = (-1.25594-888.1232/Mw) ^ (-1) 

    # --- Entropy and reduced viscosity ---
    s_res = entropy_res(model, P, T)
    m_gc = sum(model.groups.n_groups[1])
    R = Rgas()
    z = s_res / (R * m_gc)  # molar entropy term

    viscosity_reduced = exp(A + B * z + C * z^2 + D * z^3)

    # --- Combine with dilute gas viscosity (use your IB_CE function) ---
    viscosity_CE = IB_CE(model, T)
    viscosity = viscosity_reduced * viscosity_CE

    return viscosity
end


=#

function Lotgering_dilute_gas_viscosity(model::EoSModel,T)
    """
    Chapman-Enskog Theory for two component because way sigma specified
    Replace with one component something
    """
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    σ = σ_OFE(model)
    Mw = Clapeyron.molecular_weight(model) # in kg/mol
    #m_gc = sum(model.groups.n_groups[1])
    Ω = Ω⃰(model,T)
    visc = 5/16*sqrt(Mw*k_B*T/(N_A*pi))/(σ^2*Ω)
    return visc
end

