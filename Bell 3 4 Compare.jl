using Plots, LaTeXStrings, CSV, DataFrames, Clapeyron

#4 param
include("bell_functions.jl")
#L = 1000
#s_xi = LinRange(0, 30, L)

#n_g_3_og = [0.30136975, -0.11931025, 0.02531175]

#n_exp = [1.8, 2.4, 2.8]

#n_g_3_og = [0.7918253, -0.4792172, 0.06225174]
#n_exp = [ 1.76, 2.134, 2.608]


#ln_n_red_3_og = (n_g_3_og[1] .* (s_xi) .^ n_exp[1] + n_g_3_og[2] .* (s_xi) .^ n_exp[2] + n_g_3_og[3] .* (s_xi) .^ n_exp[3]) #+C*log(T)
#n_red_3_og = exp.(n_g_3_og[1] .* (s_xi) .^ (1.8) + n_g_3_og[2] .* (s_xi) .^ (2.4) + n_g_3_og[3] .* (s_xi) .^ (2.8)) .- 1.0


#exp_dodecane = CSV.read("Training Data/Decane DETHERM.csv", DataFrame)
#exp_dodecane = CSV.read("Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv", DataFrame)
#exp_dodecane = CSV.read("Training DATA/Branched Alkane/squalane.csv", DataFrame)
exp_dodecane = CSV.read("Training DATA/Hexadecane DETHERM.csv", DataFrame)
model = SAFTgammaMie(["Hexadecane"])

function reduced_visc(model::EoSModel, P, T, visc)

	visc_CE = IB_CE(model, T)
	R = Rgas()

	total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

	s_res = entropy_res(model, P, T)
	s_red = total_sf .*(-s_res ./ R).^1.0 	./ (log(T) .^ 1.0)

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model)
	m = Mw / N_A

	n_reduced = (visc - visc_CE) * ((s_red) .^ (2 / 3)) ./ ((ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T))
	return n_reduced
end

P = exp_dodecane[:, 1]
T = exp_dodecane[:, 2]
visc_exp = exp_dodecane[:, 3]

eta_red = reduced_visc.(model, P, T, visc_exp)

y_axis = log.(eta_red .+ 1.0)
num_groups = model.groups.n_groups[1]
xi = 1 # 0.4085265 * num_groups[1] + 0.0383325 * num_groups[2]

epsilon = ϵ_OFE(model)

x_axis_1 = (-entropy_res.(model, P, T) ./ (Rgas() .* xi)).^ 1.0

tot_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

x_axis = tot_sf .* x_axis_1 .^ 1.0 ./ (log.(T) .^ 1.0)
#x_axis = x_axis_1 .^ 1.0 ./ log.(100*T/epsilon) .^ 1.0 # for squalane
#x_axis = x_axis_1 .^ 1.0 ./ log.(100*T/epsilon) .^ 1.0
# squalane 30, hexa 100, 150-300 octane
#=
n_g_3 = [0.29136977, -0.1331025, 0.021]
exponents = [1.8, 2.3, 2.74]

ln_n_red_3 = n_g_3[1] .* (s_xi) .^ exponents[1] + n_g_3[2] .* (s_xi) .^ exponents[2] + n_g_3[3] .* (s_xi) .^ exponents[3]
=#
plot1 = scatter(x_axis, y_axis,
	grid = false,
	xlabel = L"(s^+/(\ln(30\times T/\epsilon)))",
	ylabel = L"\ln(\eta_\textrm{res}^+ +1)",
	xguidefontsize = 15.0,
	yguidefontsize = 15.0,
	title = "Squalane Dimensionless Experimental Data",
	lw = :3,
	marker = :diamond,
	label = false,
	xlims = (minimum(x_axis), maximum(x_axis)),
	ylims = (minimum(y_axis), maximum(y_axis)))
#=
plot1 = plot(s_xi, ln_n_red_3_og,
	grid = false,
	lw = :3,
	label = false,
	xlabel = L"(s^+/(\ln(30\times T/\epsilon)))",
	#xlabel = L"s^+",
	ylabel = L"\ln(\eta_\textrm{res}^+ +1)",
	xguidefontsize = 15.0,
	yguidefontsize = 15.0,
	title = "Squalane Dimensionless Experimental Data")
scatter!(plot1, x_axis, y_axis,
	lw = :3,
	marker = :diamond,
	label = false,
	xlims = (minimum(x_axis), maximum(x_axis)),
	ylims = (minimum(y_axis), maximum(y_axis)))
=#
#savefig(plot1,"Squalane, plus scaled no Temp")