using Plots, LaTeXStrings, CSV, DataFrames, Clapeyron

#4 param
include("bell_functions.jl")
L = 1000
s_xi = LinRange(0, 3, L)

n_g_3_og = [1.604172, -1.941327, 1.070956]

n_exp = [1.8, 2.4, 2.8]

#n_g_3_og = [0.7918253, -0.4792172, 0.06225174]
#n_exp = [ 1.76, 2.134, 2.608]


ln_n_red_3_og = (n_g_3_og[1] .* (s_xi) .^ n_exp[1] + n_g_3_og[2] .* (s_xi) .^ n_exp[2] + n_g_3_og[3] .* (s_xi) .^ n_exp[3]) #+C*log(T)


#n_red_3_og = exp.(n_g_3_og[1] .* (s_xi) .^ (1.8) + n_g_3_og[2] .* (s_xi) .^ (2.4) + n_g_3_og[3] .* (s_xi) .^ (2.8)) .- 1.0

#exp_dodecane = CSV.read("Training Data/ethane.csv", DataFrame)
exp_dodecane = CSV.read("Training Data/Hexane DETHERM.csv", DataFrame)
#exp_dodecane = CSV.read("Training DATA/Branched Alkane/2,6,10,14-tetramethylpentadecane.csv", DataFrame)
#exp_dodecane = CSV.read("Training DATA/Branched Alkane/squalane.csv", DataFrame)
#exp_dodecane = CSV.read("Validation DATA/Heptadecane DETHERM.csv", DataFrame)
model = SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal)

function reduced_visc(model::EoSModel, P, T, visc)

	visc_CE = IB_CE(model, T)
	R = Rgas()

	total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

	s_res = entropy_res(model, P, T)
	s_id = Clapeyron.entropy(model, P, T) - entropy_res(model, P, T)

	crit_point = crit_pure(model)
	s_crit = entropy_res(model, crit_point[2], crit_point[1])
	s_red = -s_res/s_id# + 1.0 * log(s_res/s_crit)
	#s_red = s_res/s_crit

	epsilon_ofe = ϵ_OFE(model)

	#s_red = (-s_res ./ R).^1.0 	#./ (log(T) .^ 1.0)

	N_A = Clapeyron.N_A
	k_B = Clapeyron.k_B

	ρ_molar = molar_density(model, P, T)
	ρ_N = ρ_molar .* N_A

	Mw = Clapeyron.molecular_weight(model)
	m = Mw / N_A

	n_reduced = (visc - visc_CE)  ./ ((ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) #* ((s_red) .^ (2 / 3))
	#n_reduced = (visc - visc_CE) *((s_red) .^ (2 / 3))
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

#x_axis_1 = (-entropy_res.(model, P, T) ./ (Rgas() .* xi)).^ 1.0
crit_point = crit_pure(model)
s_crit = entropy_res(model, crit_point[2], crit_point[1])
s_ideal = (Clapeyron.entropy.(model, P, T).- entropy_res.(model,P,T))
s_resid = entropy_res.(model, P, T)

x_axis_1 = -s_resid ./ (Rgas() .* ( 1 .- 0.161 .* sqrt.(T./crit_point[1])))
#x_axis_1 = entropy_res.(model, P, T)./s_crit

tot_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

x_axis = x_axis_1 .^ 1.0 #./ (log.(T) .^ 1.0)
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
	xlabel = L"s^+ \times (1 + m \sqrt{(T/ T_C)})",
	ylabel = L"\ln(\eta_\textrm{reduced} +1)",
	xguidefontsize = 15.0,
	yguidefontsize = 15.0,
	title = "Hexane Dimensionless Experimental Data",
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
#savefig(plot1,"Heptadecane, plus scaled Temp")

# Define the models you want to compare

models = [
#    SAFTgammaMie(["Pentane"], idealmodel = WalkerIdeal),
#    SAFTgammaMie(["Hexane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Octane"], idealmodel = WalkerIdeal),
    SAFTgammaMie(["Decane"], idealmodel = WalkerIdeal),
 #   SAFTgammaMie(["Dodecane"], idealmodel = WalkerIdeal),
#    SAFTgammaMie(["Tridecane"], idealmodel = WalkerIdeal),
#    SAFTgammaMie(["Hexadecane"], idealmodel = WalkerIdeal),
#  	SAFTgammaMie(["Eicosane"], idealmodel = WalkerIdeal),

]

# models = [
#     SAFTgammaMie(["Pentane"], idealmodel = BasicIdeal),
#     SAFTgammaMie(["Hexane"], idealmodel = BasicIdeal),
#     SAFTgammaMie(["Octane"], idealmodel = BasicIdeal),
#     SAFTgammaMie(["Decane"], idealmodel = BasicIdeal),
#     SAFTgammaMie(["Dodecane"], idealmodel = BasicIdeal),
#     SAFTgammaMie(["Tridecane"], idealmodel = BasicIdeal),
#     SAFTgammaMie(["Hexadecane"], idealmodel = BasicIdeal),
# 	SAFTgammaMie(["Eicosane"], idealmodel = BasicIdeal),
# 	SAFTgammaMie(["Squalane"], idealmodel = BasicIdeal)
# ]


# Matching CSV file names (must exist)
data_files = [
    "Training Data/Pentane DETHERM.csv",
    "Training Data/Hexane DETHERM.csv",
    "Training Data/Octane DETHERM.csv",
    "Training Data/Decane DETHERM.csv",
    "Training Data/Dodecane DETHERM.csv",
    "Validation Data/Tridecane DETHERM.csv",
    "Training Data/Hexadecane DETHERM.csv",
	"Training DATA/n-eicosane.csv",
	"Training DATA/Branched Alkane/squalane.csv"
]

# Function for reduced viscosity
function reduced_visc(model::EoSModel, P, T, visc)
	xi_ch3 = 0.2
	xi_ch2 = 0.004

	num_groups_comp = model.groups.n_groups[1]
	xi = num_groups_comp[1] * xi_ch3 + num_groups_comp[2] * xi_ch2 

    visc_CE = IB_CE(model, T)
    R = Rgas()
    total_sf = sum(model.params.shapefactor.values .* model.groups.n_groups[1])
    s_res = entropy_res(model, P, T)
    s_id = Clapeyron.entropy(model, P, T) .- s_res
    crit_point = crit_pure(model)
    s_crit = entropy_res(model, crit_point[2], crit_point[1])
    s_red = (-s_res ./ s_id .+  log.(s_res ./ s_crit))
	#s_red = -s_res ./ sum(num_groups)
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A
    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A
    n_reduced = (visc - visc_CE) ./ ((ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) .*  ((s_red) .^ (2 / 3))
    return n_reduced
end

# Initialize plot
plotall = plot(
    grid = false,
    xlabel = L"s^+",
    ylabel = L"\ln(\eta_\mathrm{res}^+ + 1)",
    title = "Dimensionless Experimental Data for Linear Alkanes",
    xguidefontsize = 15.0,
    yguidefontsize = 15.0,
    lw = 3,
    legend = :topright
)

# Loop through all models and datasets
xi_ch3 = 0.2
xi_ch2 = 0.004
for (model, file) in zip(models, data_files)
    exp_data = CSV.read(file, DataFrame)
    P = exp_data[:, 1]
    T = exp_data[:, 2]
    visc_exp = exp_data[:, 3]

    eta_red = reduced_visc.(model, P, T, visc_exp)

	num_groups_comp = model.groups.n_groups[1]
	xi = num_groups_comp[1] * xi_ch3 + num_groups_comp[2] * xi_ch2 

    s_res = entropy_res.(model, P, T)
    s_id = Clapeyron.entropy.(model, P, T) .- s_res
    crit_point = crit_pure(model)
    s_crit = entropy_res(model, crit_point[2], crit_point[1])

    
	total_sf_comp = sum(model.params.shapefactor.values .* model.groups.n_groups[1])

	x_axis = (-s_res ./ s_id .+   log.(s_res ./ s_crit)) 

	#x_axis = -s_res ./ sum(num_groups_comp)
    y_axis = log.(eta_red .+ 1.0)

    comp_name = model.groups.components
    scatter!(plotall, x_axis, y_axis, label = false, marker = :auto)
	#print(x_axis, y_axis)
end

display(plotall)
plot!(plotall,s_xi,ln_n_red_3_og, lw = :3)

a1, a2, b1, b2, c, d = 1.83125641, -0.00653110, 2.07866481, 0.22647604, -1.48457090, 0.89861164
#a1, a2, b1, b2, c, d = −0.35918064, 0.00488280, 0.59431770, −0.05315258, 0.44624897, 0.41245087
model = SAFTgammaMie(["Decane"], idealmodel = WalkerIdeal)
exp_data = CSV.read("Training Data/Decane DETHERM.csv",DataFrame)
R = Rgas()
ent_res = entropy_res.(model,exp_data[:,1], exp_data[:,2])
s_id = Clapeyron.entropy.(model, exp_data[:,1], exp_data[:,2]) .- ent_res
crit_point = crit_pure(model)
s_crit = entropy_res(model, crit_point[2], crit_point[1])
x_es = (-ent_res ./ s_id .+   log.(ent_res ./ s_crit)) 


ln_n_nref = ((a1 .+ a2 .* ent_res)./(1 .+ exp.(c .* x_es)) .+ (b1 .+ b2 .* ent_res)./ (1 .+ exp.(-c .* x_es))) .* x_es .+ d./s_crit
#ln_n_nref = (a1 .+ a2 .* ent_res./R)./(1 .+ exp.(c .* x_es))  .* x_es .+ d./(s_crit./R)
plot!(plotall, x_es, ln_n_nref)
# savefig(plot1, "All_alkanes_dimensionless_viscosity.png")

