using Clapeyron, PyCall, Plots

model_water = SAFTgammaMie(["Water"])

function p_T(model)
	(Tc, pc, vc) = crit_pure(model)
	N = 100

	T    = LinRange(298.15, Tc, N)
	psat = zeros(N)
	vl   = zeros(N)
	vv   = zeros(N)

	hL  = zeros(N)
	hV  = zeros(N)
	cpL = zeros(N)
	cpV = zeros(N)

	for i in 1:N
		if i == 1
			sat = saturation_pressure(model, T[i])
			psat[i] = sat[1]
			vl[i] = sat[2]
			vv[i] = sat[3]
			v0 = [vl[i], vv[i]]
		else
			sat = saturation_pressure(model, T[i]; v0 = v0)
			psat[i] = sat[1]
			vl[i] = sat[2]
			vv[i] = sat[3]
			v0 = [vl[i], vv[i]]
		end
		hL[i]  = Clapeyron.VT_enthalpy(model, vl[i], T[i], [1.0])
		hV[i]  = Clapeyron.VT_enthalpy(model, vv[i], T[i], [1.0])
		cpL[i] = Clapeyron.VT_isobaric_heat_capacity(model, vl[i], T[i], [1.0])
		cpV[i] = Clapeyron.VT_isobaric_heat_capacity(model, vv[i], T[i], [1.0])
	end
    plt = plot(grid=:off,framestyle=:box,foreground_color_legend = nothing,legend_font=font(12))
    plot!(plt,T,psat,color="red",line = (:path, 3),label = false)

    #ylabel!(plt,L"Temperature [K]",xguidefontsize=12)
    #xlabel!(plt,L"Density [kgm^{-3}]",yguidefontsize=12)
end
