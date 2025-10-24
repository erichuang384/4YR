using Plots, LaTeXStrings

#4 param
L = 1000
s_xi = LinRange(2,20,L)

n_g_3_og = [0.30136975, -0.11931025, 0.02531175]

ln_n_red_3_og  = n_g_3_og[1] .* (s_xi) .^ (1.8) + n_g_3_og[2] .* (s_xi) .^ (2.4) + n_g_3_og[3] .* (s_xi) .^ (2.8)

#n_g_3 = [0.281, -0.105, 0.0258]
#exponents = [1.7, 2.2, 2.5]

#n_g_3 = [0.2, -0.08, 0.045]
#exponents = [1.7, 2.3, 2.4]

n_g_3 = [0.30136975, -0.12931025, 0.03631175]
exponents = [1.8, 2.4, 2.7]

ln_n_red_3 = n_g_3[1] .* (s_xi) .^ exponents[1] + n_g_3[2] .* (s_xi) .^ exponents[2] + n_g_3[3] .* (s_xi) .^ exponents[3]


plot1 = plot(s_xi,ln_n_red_3_og,
grid = false,
lw =:3,
label = "3 param original",
xlabel = L"s^+ / \xi_\textrm{exp}",
ylabel = L"ln(\eta_\textrm{res}^+ +1)")
plot!(plot1,s_xi,ln_n_red_3,
lw=:3,
label = "3 param")