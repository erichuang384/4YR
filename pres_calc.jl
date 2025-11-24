using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings, StaticArrays
using CMAEvolutionStrategy
using Random, Statistics

# -----------------------------------
# Build model and experimental data
# -----------------------------------
model = SAFTgammaMie(["Hexane","Octane"])  # SAFT-γ Mie for a binary mixture

# Experimental hexane mole fractions
x_hexane = [0.1042, 0.1995, 0.2959, 0.3973, 0.4949, 0.5975, 0.6917, 0.7940, 0.8955]
z = hcat(x_hexane, 1 .- x_hexane)  # compositions (Hexane, Octane)

T_ref = 293.15  # K

# Convert exp density [kg/L] to molar volume [m^3/mol]
exp_rho = [0.6995, 0.6965, 0.6929, 0.6891, 0.6852, 0.6809, 0.6767, 0.6720, 0.6668] # kg/L (example single-density value, adjust if you have per-z densities)
Mw = [Clapeyron.molecular_weight(model, z[i, :]) for i in 1:size(z,1)] # g/mol
# ρ [kg/L] -> ρ [kg/m^3] -> ρ_mol [mol/m^3] = ρ [kg/m^3] / (Mw [kg/mol])
exp_rho_mol = exp_rho .* 1e3./ (Mw./ 1000)       # mol/m^3
exp_vol_mol = 1.0./ exp_rho_mol                   # m^3/mol (target volumes)

# Sanity: pressure from EOS that corresponds to each exp_vol_mol
exp_pres = zeros(length(exp_vol_mol))
for i in eachindex(exp_pres)
    exp_pres[i] = Clapeyron.pressure(model, exp_vol_mol[i], T_ref, z[i,:])  # Pa
end

println("Sample: p(z₁) from EOS at experimental molar volume: ", exp_pres[1], " Pa")
println("volume(model, p, T, z) at p=exp_pres[1]: ",
        volume(model, exp_pres[1], T_ref, z[1,:]), " m^3/mol")
println("Target exp_vol_mol[1]: ", exp_vol_mol[1], " m^3/mol")

# -----------------------------------
# MODE A — Fit per-composition pressures to reproduce exp_vol_mol
#         (sanity: CMA-ES will converge to Clapeyron.pressure outputs)
# -----------------------------------
function sse_pressures(pvec::AbstractVector{<:Real})
    # Penalize negative pressures
    if any(pvec.<= 0)
        return 1e20
    end
    total = 0.0
    for i in eachindex(pvec)
        v_pred = volume(model, pvec[i], T_ref, z[i,:])   # m^3/mol
        v_exp  = exp_vol_mol[i]
        total += ((v_pred - v_exp) / v_exp)^2
    end
    return total / length(pvec)
end

# Initialize at EOS-implied pressures (this is essentially a direct solution)
x0_pressures = fill(1e5,9)
σ0_pressures = 0.1  # initial CMA-ES step
Random.seed!(123)

println("\nStarting CMA-ES fit of per-composition pressures...")
julia
# -----------------------------------
# MODE A — Fit per-composition pressures to reproduce exp_vol_mol
#         (with CMA-ES bounds: 0 ≤ p_i ≤ 1e8 Pa)
# -----------------------------------
function sse_pressures(pvec::AbstractVector{<:Real})
    # Penalize non-physical pressures (CMA-ES also enforces bounds, but we guard here)
    if any(pvec.< 0.0) || any(pvec.> 1.0e8)
        return 1e20
    end
    total = 0.0
    for i in eachindex(pvec)
        v_pred = volume(model, pvec[i], T_ref, z[i,:])   # m^3/mol
        v_exp  = exp_vol_mol[i]
        total += ((v_pred - v_exp) / v_exp)^2
    end
    return total / length(pvec)
end

# Initialize at EOS-implied pressures (direct solution)
x0_pressures = fill(1e5,length(exp_rho))

# CMA-ES step size
σ0_pressures = 0.1

# Bounds: 0 to 1e8 Pa for each composition
lower_pressures = fill(0.0, length(x0_pressures))
upper_pressures = fill(1.0e8, length(x0_pressures))

Random.seed!(123)
println("\nStarting CMA-ES fit of per-composition pressures with bounds [0, 1e8] Pa...")
res_press = minimize(
    sse_pressures,
    x0_pressures,
    σ0_pressures;
    lower = lower_pressures,
    upper = upper_pressures,
    seed = 123,
    maxiter = 1000,
    ftol = 1e-12,
    verbosity = 1
)
p_best = xbest(res_press)
sse_best_press = fbest(res_press)
println("✅ CMA-ES (pressures, bounded) done. SSE = ", sse_best_press)
println("First few fitted pressures [Pa]: ", p_best[1:min(end,3)])
p_best = xbest(res_press)
sse_best_press = fbest(res_press)
println("✅ CMA-ES (pressures) done. SSE = ", sse_best_press)
println("First few fitted pressures [Pa]: ", p_best[1:min(end,3)])

# Check reconstruction accuracy
v_pred_press = [volume(model, p_best[i], T_ref, z[i,:]) for i in eachindex(p_best)]
@show maximum(abs.(v_pred_press.- exp_vol_mol))

# -----------------------------------
# MODE B — Fit a single binary interaction parameter k_ij for Hexane–Octane
#          to match exp_vol_mol at a fixed reference pressure (e.g., 1 bar)
# -----------------------------------

# Helper: set (or update) binary interaction "k" in the model.
# NOTE: This assumes your model has params.k.values (common for SAFT-γ Mie).
#       If your Clapeyron version differs, update the setter accordingly.
function set_kij(m::EoSModel, k::Real)
    m2 = deepcopy(m)
    # Guard: check if the parameter exists
    if hasproperty(m2.params, :k)
        # Set symmetric k_{12} = k_{21} = k
        m2.params.k.values[1,2] = k
        m2.params.k.values[2,1] = k
    else
        @warn "Model has no 'k' parameter; cannot set k_ij. Returning original model."
    end
    return m2
end

# Objective: SSE over molar volumes at a fixed pressure p_ref (e.g., 1 bar)
p_ref = 1.0e5 # Pa (adjust if you know the experimental pressure)
function sse_kij(k::AbstractVector{<:Real})
    k12 = k[1]
    m_k = set_kij(model, k12)
    total = 0.0
    for i in 1:size(z,1)
        v_pred = volume(m_k, p_ref, T_ref, z[i,:])
        v_exp  = exp_vol_mol[i]
        # robust relative SSE
        total += ((v_pred - v_exp) / v_exp)^2
    end
    return total / size(z,1)
end

# CMA-ES on a single scalar k_ij
x0_k = [0.0]         # start at no interaction correction
σ0_k = 0.05          # initial step
lower_k = [-0.5]     # reasonable range for SAFT-γ Mie kij
upper_k = [0.5]

println("\nStarting CMA-ES fit of binary interaction k_ij (Hexane–Octane)...")
res_k = minimize(
    sse_kij,
    x0_k,
    σ0_k;
    lower = lower_k,
    upper = upper_k,
    seed = 456,
    maxiter = 3000,
    ftol = 1e-12,
    verbosity = 1
)
k_best = xbest(res_k)[1]
sse_best_k = fbest(res_k)
println("✅ CMA-ES (k_ij) done. SSE = ", sse_best_k)
println(@sprintf("Best k_ij(Hexane–Octane) = %.6f", k_best))

# Compare predicted vs experimental molar volumes with best k_ij at p_ref
model_kbest = set_kij(model, k_best)
v_pred_k = [volume(model_kbest, p_ref, T_ref, z[i,:]) for i in 1:size(z,1)]

# -----------------------------------
# Plot: experimental vs predicted molar volumes (k_ij fit)
# -----------------------------------
plot1 = plot(
    z[:,1], exp_vol_mol,
    marker=:circle, label="Experimental molar volume",
    xlabel="Hexane mole fraction",
    ylabel="Molar volume [m^3/mol]",
    title="Molar volume at $(p_ref/1e5) bar, T=$(T_ref) K",
    grid=true, lw=2
)
plot!(z[:,1], v_pred_k, marker=:square, label="Predicted (best k_ij)", lw=2)
display(plot1)

# -----------------------------------
# Optional: report pressures that match exp_vol_mol (sanity check)
# -----------------------------------
println("\nSanity check pressures from EOS for each composition (at exp_vol_mol):")
for i in 1:size(z,1)
    println(@sprintf("x_hexane=%.4f -> p=%.3f bar", z[i,1], exp_pres[i]/1e5))
end

# -----------------------------------
# Summary
# -----------------------------------
println("\nSummary:")
println(@sprintf("Best k_ij fit SSE (relative): %.6e", sse_best_k))
println(@sprintf("Best per-composition pressure fit SSE: %.6e", sse_best_press))
println(@sprintf("Max |v_pred(p_best) - v_exp| = %.6e m^3/mol", maximum(abs.(v_pred_press.- exp_vol_mol))))