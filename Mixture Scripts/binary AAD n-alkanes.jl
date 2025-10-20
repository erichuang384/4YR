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

# calculate viscosity from Bell
function calc_AAD(model::EoSModel, exp_file,rule = 1)
    """
    calculates viscosity from Bell method, returns x_vals and viscosities 
    """
    p = exp_file[:,1]
    T = exp_file[:,2]
    x = exp_file[:,3]
    visc_exp = exp_file[:,4]
    visc = zeros(length(x))
    for i in 1:length(x)
        visc[i] = IB_viscosity_mix(model, p[i], T[i], [x[i], 1.0 - x[i]],rule = rule)
    end
    AAD = sum(abs.(visc_exp-visc)./visc_exp)/length(x)
    return AAD
end

#
AAD_all = calc_AAD.(models,exp_data)
mean_AAD = sum(AAD_all)/length(AAD_all)

#
AAD_all_2 = calc_AAD.(models,exp_data,2)
mean_AAD_2 = sum(AAD_all_2)/length(AAD_all_2)


function IB_viscosity_mix(model::EoSModel, P, T, z = StaticArrays.SA[1.0];rule = 1)
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
    if rule == 1
	    ξ_mix = sum(z .* ξ_pure) #molecular weighted
    elseif rule == 2
        ξ_mix = sum(z ./ ξ_pure)^(-1)
    end
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
