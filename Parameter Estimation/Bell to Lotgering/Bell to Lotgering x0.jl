
using Clapeyron
using Polynomials, Plots

# Model setup
models = SAFTgammaMie.([["Butane"], ["Pentane"], ["Hexane"], ["Heptane"],
    ["Octane"], ["Nonane"], ["Decane"], ["Undecane"], ["Dodecane"],
    ["Tridecane"], ["Tetradecane"], ["Pentadecane"], ["Hexadecane"], ["Heptadecane"]])


# Arbitrary conditions
P = 1e5
T = zeros(length(models))
for i in 1:length(models)
    T_boil = saturation_temperature(models[i],1e5)
    T[i] = T_boil[1] - 10
end

N_A = Clapeyron.N_A
k_B = Clapeyron.k_B

Mw = Clapeyron.molecular_weight.(models)
m = Mw./ N_A

ρ_molar = molar_density.(models, P, T)
ρ_N = ρ_molar .* Clapeyron.N_A

#constant = (ρ_N^(2/3)) * sqrt(m * k_B * T) / IB_CE(model, T)

# Bell coefficients
n_g_all = [
    0.369686 −0.148651 0.032041;    # Butane
    0.404638 −0.184815 0.042214;    # Pentane
    0.355662 −0.151115 0.032927;    #...
    0.321695 −0.130973 0.027822;    #Hept
    0.327911 −0.138181 0.029726;    # oct
    0.35713  −0.159116 0.034753;# nonane
    0.306021 −0.125138 0.02603; # decane
    0.298732 −0.119667 0.024296;
    0.26995  −0.10285 0.020241;
    0.286312 −0.111090 0.022642; # global
    0.286312 −0.111090 0.022642; # global
    0.286312 −0.111090 0.022642;# global
    0.236788 −0.085333 0.015984;
    0.286312 −0.111090 0.022642 # global
]

# Test curve
N = 1000
s = LinRange(0, 15, N)
#y = constant .* (n_g[1] .* s .^ 1.8 .+ n_g[2] .* s .^ 2.4 .+ n_g[3] .* s .^ 2.8) .- constant .+ 1

bell_curve(s, n_g) = n_g[1] .* s .^ 1.8 .+ n_g[2] .* s .^ 2.4 .+ n_g[3] .* s .^ 2.8
# === Polynomial fitting section ===
fits = [Polynomials.fit(Polynomial, s, bell_curve(s, n_g_all[i, :]), 3) for i in 1:length(models)]

#Plot functions
plots = Vector{Plots.Plot{Plots.GRBackend}}(undef, length(models))  

for i in 1:length(models)
    y = bell_curve(s, n_g_all[i, :])
    y_fit = fits[i].(s)
    name = models[i].components[1]

    p = plot(
        s, y, label="Original", lw=3,
        xlabel="s", ylabel="y(s)",
        title="Quartic fit for $(name)"
    )
    plot!(p, s, y_fit, label="Quartic fit", lw=2, ls=:dot)
    plots[i] = p
end

using DataFrames, CSV

# Extract coefficient data from each fit
# Note: coeffs(p) gives coefficients in ascending order: a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
coeff_data = [
    (
        Component = models[i].components[1],
        a0 = coeffs(fits[i])[1],
        a1 = coeffs(fits[i])[2],
        a2 = coeffs(fits[i])[3],
        a3 = coeffs(fits[i])[4]
    )
    for i in 1:length(models)
]

# Convert to DataFrame
df = DataFrame(coeff_data)

# Write to CSV
CSV.write("quartic_fits.csv", df)

