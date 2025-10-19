using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, StaticArrays
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))

#models
models = [SAFTgammaMie(["Decane", "Pentadecane"]),
        SAFTgammaMie(["Decane", "Tridecane"]),
      SAFTgammaMie(["Octane", "Pentadecane"]),
   SAFTgammaMie(["Octane", "Tridecane"]),
    SAFTgammaMie(["Octane", "Undecane"]),
    SAFTgammaMie(["Tridecane", "Pentadecane"]),
     SAFTgammaMie(["Undecane", "Pentadecane"]),
     SAFTgammaMie(["Undecane", "Tridecane"]),
]

#Format is p,T,x,viscosity

exp_data = [CSV.read("Experimental Data/Binary Liquid n-alkane/decane_pentadecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/decane_tridecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/octane_pentadecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/octane_tridecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/octane_undecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/tridecane_pentadecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/undecane_pentadecane.csv", DataFrame),
    CSV.read("Experimental Data/Binary Liquid n-alkane/undecane_tridecane.csv", DataFrame),
]

p_exp = exp_octane_undecane[:, 1]
T_exp = exp_octane_undecane[:, 2]
x_exp = exp_octane_undecane[:, 3]
z_exp = hcat(x_exp, 1 .- x_exp) #
visc_exp = exp_octane_undecane[:, 4]


# calculate viscosity from Bell
function calc_visc(model::EoSModel, p, T, rule = 1)
    """
    calculates viscosity from Bell method, returns x_vals and viscosities 
    """
    n = 200
    x = LinRange(0, 1, n)
    visc = zeros(length(x))
    for i in 1:length(x)
        visc[i] = IB_viscosity_mix_test(model, p, T, [x[i], 1.0 - x[i]],rule = rule)
    end
    return visc
end

#In this data set have values at 293.15K and 298.15K
visc_calc_293 = calc_visc.(models,101325,293.15)
visc_calc_293_2 = calc_visc.(models,101325,293.15,2)
visc_calc_298 = calc_visc.(models,101325,298.15)

# plot experimental Data
x_range = LinRange(0,1,200)
#=
plot_x1_decane = scatter(exp_data[1][1:11,3], exp_data[1][1:11,4],
	grid = false,
	label = false,
    color =:blue,
    marker =:diamond,
	xlabel = L"x",
	ylabel = L"\eta/\textrm{Pa s}")
    scatter!(plot_x1_decane,exp_data[1][12:end,3], exp_data[1][12:end,4],
    marker =:diamond,
    color =:red,
    label = false)
    plot!(plot_x1_decane,x_range,visc_calc_293[1],
    color =:blue,
    label = false)
    plot!(plot_x1_decane,x_range,visc_calc_298[1],
    color=:red,
    label = false)
=#
plot_x1_decane_293 = scatter(exp_data[1][1:11,3], exp_data[1][1:11,4],# decane pentadecane
	grid = false,
	label = "pentadecane",
    color =:blue,
    marker =:diamond,
	xlabel = L"x_\textrm{n-decane}",
	ylabel = L"\eta/\textrm{Pa s}")
    plot!(plot_x1_decane_293,x_range,visc_calc_293[1],
    color =:blue,
    label = false)
    #decane - tridecane
    scatter!(plot_x1_decane_293,exp_data[2][1:11,3], exp_data[2][1:11,4],
	label = "tridecane",
    color =:red,
    marker =:circle)
    plot!(plot_x1_decane_293,x_range,visc_calc_293[2],
    color =:red,
    label = false)
xlims!(plot_x1_decane_293,0,1)

plot_x1_octane_293 = scatter(exp_data[3][1:11,3], exp_data[3][1:11,4],
	grid = false,
	label = "pentadecane",
    color =:blue,
    marker =:diamond,
	xlabel = L"x_\textrm{n-Octane}",
	ylabel = L"\eta/\textrm{Pa s}")
    plot!(plot_x1_octane_293,x_range,visc_calc_293[3],
    color =:blue,
    label = false)
    # octane - pentadecane
    scatter!(plot_x1_octane_293,exp_data[4][1:11,3], exp_data[4][1:11,4],
	label = "tridecane",
    color =:red,
    marker =:circle)
    plot!(plot_x1_octane_293,x_range,visc_calc_293[4],
    color =:red,
    label = false)
    # octane - pentadecane
    scatter!(plot_x1_octane_293,exp_data[5][1:11,3], exp_data[5][1:11,4],
	label = "undecane",
    color =:green,
    marker =:square)
    plot!(plot_x1_octane_293,x_range,visc_calc_293[5],
    color =:green,
    label = false)
xlims!(plot_x1_octane_293,0,1)

plot_x1_undecane_293 = scatter(exp_data[7][1:11,3], exp_data[7][1:11,4], # undecane - pentadecane
	grid = false,
	label = "pentadecane",
    color =:blue,
    marker =:diamond,
	xlabel = L"x_\textrm{n-undecane}",
	ylabel = L"\eta/\textrm{Pa s}")
    plot!(plot_x1_undecane_293,x_range,visc_calc_293[7],
    color =:blue,
    label = false)
    # undecane - tridecane
    scatter!(plot_x1_undecane_293,exp_data[8][1:11,3], exp_data[8][1:11,4],
	label = "tridecane",
    color =:red,
    marker =:circle)
    plot!(plot_x1_undecane_293,x_range,visc_calc_293[8],
    color =:red,
    label = false)
xlims!(plot_x1_undecane_293,0,1)

savefig(plot_x1_decane_293,"Decane 293")
savefig(plot_x1_octane_293,"Octane 293")
savefig(plot_x1_undecane_293,"Undecane 293")

plot_x1_octane_298 = scatter(exp_data[3][12:end,3], exp_data[3][12:end,4],
	grid = false,
	label = "pentadecane",
    color = :blue,
    marker = :diamond,
	xlabel = L"x_\textrm{n-Octane}",
	ylabel = L"\eta/\textrm{Pa s}")
plot!(plot_x1_octane_298, x_range, visc_calc_298[3],
    color = :blue,
    label = false)

# octane - tridecane
scatter!(plot_x1_octane_298, exp_data[4][12:end,3], exp_data[4][12:end,4],
	label = "tridecane",
    color = :red,
    marker = :circle)
plot!(plot_x1_octane_298, x_range, visc_calc_298[4],
    color = :red,
    label = false)

# octane - undecane
scatter!(plot_x1_octane_298, exp_data[5][12:end,3], exp_data[5][12:end,4],
	label = "undecane",
    color = :green,
    marker = :square)
plot!(plot_x1_octane_298, x_range, visc_calc_298[5],
    color = :green,
    label = false)
xlims!(plot_x1_octane_298, 0, 1)

# -----------------------------------------------------

plot_x1_undecane_298 = scatter(exp_data[7][12:end,3], exp_data[7][12:end,4],  # undecane - pentadecane
	grid = false,
	label = "pentadecane",
    color = :blue,
    marker = :diamond,
	xlabel = L"x_\textrm{n-undecane}",
	ylabel = L"\eta/\textrm{Pa s}")
plot!(plot_x1_undecane_298, x_range, visc_calc_298[7],
    color = :blue,
    label = false)

# undecane - tridecane
scatter!(plot_x1_undecane_298, exp_data[8][12:end,3], exp_data[8][12:end,4],
	label = "tridecane",
    color = :red,
    marker = :circle)
plot!(plot_x1_undecane_298, x_range, visc_calc_298[8],
    color = :red,
    label = false)

xlims!(plot_x1_undecane_298, 0, 1)

savefig(plot_x1_octane_298,"Octane 298")
savefig(plot_x1_undecane_298,"undecane 298")


# plot different mixing rules on 293 octane
plot_x1_octane_293_mix = scatter(exp_data[3][1:11,3], exp_data[3][1:11,4],
	grid = false,
	label = "pentadecane",
    color =:blue,
    marker =:diamond,
	xlabel = L"x_\textrm{n-Octane}",
	ylabel = L"\eta/\textrm{Pa s}")
    plot!(plot_x1_octane_293_mix,x_range,visc_calc_293[3],
    color =:blue,
    label = false)
    # octane - pentadecane
    scatter!(plot_x1_octane_293_mix,exp_data[4][1:11,3], exp_data[4][1:11,4],
	label = "tridecane",
    color =:red,
    marker =:circle)
    plot!(plot_x1_octane_293_mix,x_range,visc_calc_293[4],
    color =:red,
    label = false)
    # octane - pentadecane
    scatter!(plot_x1_octane_293_mix,exp_data[5][1:11,3], exp_data[5][1:11,4],
	label = "undecane",
    color =:green,
    marker =:square)
    plot!(plot_x1_octane_293_mix,x_range,visc_calc_293[5],
    color =:green,
    label = false)
    # second mixing rule
    plot!(plot_x1_octane_293_mix,x_range,visc_calc_293_2[3],
    color =:blue,
    ls =:dash,
    label = false)
    # octane - pentadecane
    plot!(plot_x1_octane_293_mix,x_range,visc_calc_293_2[4],
    color =:red,
    ls =:dash,
    label = false)
    # octane - pentadecane
    plot!(plot_x1_octane_293_mix,x_range,visc_calc_293_2[5],
    color =:green,
    ls =:dash,
    label = false)
savefig(plot_x1_octane_293_mix,"Octane 293 mix")

function IB_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
	"""
	Overall Viscosity using method proposed by Ian Bell
	"""
	n_g = [-0.448046, 1.012681, -0.381869, 0.054674] # global parameters
	ξ_pure = zeros(length(z))

	for j ∈ 1:length(z)

		ξ_i = ["CH3" 0.484458;
			"CH2"  0.047793] # temporary manually tuned
		ξ = 0
		# GCM determination of ξ, doesn't yet include second order contributions
		groups = model.groups.groups[j] #n-elemnet Vector{string}
		num_groups = model.groups.n_groups[j] #n-element Vector{Int}
		for i in 1:length(groups)
			value = ξ_i[ξ_i[:, 1].==groups[i], 2][1]
			ξ = ξ + value * num_groups[i]

			ξ_pure[j] = ξ
		end

	end

	ξ_mix = sum(z .* ξ_pure)
	R = Rgas()
	s_res = entropy_res(model, P, T, z)
	s_red = -s_res ./ R

	n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) + n_g[2] .* (s_red ./ ξ_mix) .^ (1.5) + n_g[3] .* (s_red ./ ξ_mix) .^ (2) + n_g[4] .* (s_red ./ ξ_mix) .^ (2.5)) - 1

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T, z)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model, z)
	m = Mw / N_A

	n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ ((s_red) .^ (2 / 3))

	if length(z) == 1
		viscosity = IB_CE(model, T) + n_res
	else
		viscosity = IB_CE_mix(model, T, z) + n_res
	end

	return viscosity
end