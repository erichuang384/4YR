using Clapeyron, LinearAlgebra

include("all_functions.jl")

model = SAFTgammaMie(["hexane","heptane","octane"])

model_a = PCPSAFT(["hexane"])
model_hexane = SAFTgammaMie(["hexane"])
#model = SAFTgammaMie(["methane","butane"])
function CE_mix(model::EoSModel,T,z)
    # need to define models
    components = model.groups.components
    models = []

    for i in 1:length(components)
        push!(models,SAFTgammaMie([components[i]]))
    end
    # calculate CE pure
    viscosity_CE = dilute_gas_viscosity.(models[:],T)
    Mw = Clapeyron.molecular_weight.(models[:])

    #Φ matrix
    n = length(viscosity_CE)
    phi = zeros(n, n)

    for i in 1:n, j in 1:n
        phi[i, j] = (1 + sqrt(viscosity_CE[i] / viscosity_CE[j]) * (Mw[j] / Mw[i])^(0.25))^2 /
        sqrt(8 * (1 + Mw[i] / Mw[j]))
    end

    phi_ij = [sum(z[j] * phi[i, j] for j in 1:n) for i in 1:n]

    visc_mix = sum(z[i] * viscosity_CE[i] / phi_ij[i] for i in 1:n)
end
z = [0.2, 0.3,0.5]

CE_mix(model,300,z)
a = model.groups.components[1]
SAFTgammaMie([a])

