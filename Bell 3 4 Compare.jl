using Plots, LaTeXStrings

#4 param
L = 1000
s_xi = LinRange(0,100,L)

n_g = [-0.448046, 1.012681, -0.381869, 0.054674]

ln_n_red_4 = n_g[1] .* (s_xi) + n_g[2] .* (s_xi) .^ (1.5) + n_g[3] .* (s_xi) .^ (2) + n_g[4] .* (s_xi) .^ (2.5)

n_g_3 = [0.310361, -0.133413, 0.032088]

ln_n_red_3 = n_g_3[1] .* (s_xi) .^ (1.8) + n_g_3[2] .* (s_xi) .^ (2.4) + n_g_3[3] .* (s_xi) .^ (2.8)


plot1 = plot(s_xi,ln_n_red_4,
grid = false,
lw =:3,
label = "4 param",
xlabel = L"s^+ / \xi_\textrm{exp}",
ylabel = L"ln(\eta_\textrm{res}^+ +1)")
plot!(plot1,s_xi,ln_n_red_3,
lw=:3,
label = "3 param")