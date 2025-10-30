using Plots, LaTeXStrings, CSV, DataFrames, Clapeyron

#4 param
include("bell_functions.jl")
L = 1000
s_xi = LinRange(0,20,L)

n_g_3_og = [0.30136975, -0.11931025, 0.02531175]

n_exp = [1.8, 2.4, 2.8]

n_g_3_og = [1.974163, -1.1569, 0.314623]
n_exp = [ 1.8, 2.4, 2.8]

#n_g_3_og = [0.7918253, -0.4792172, 0.06225174]
#n_exp = [ 1.76, 2.134, 2.608]


ln_n_red_3_og  = (n_g_3_og[1] .* (s_xi) .^ n_exp[1] + n_g_3_og[2] .* (s_xi) .^ n_exp[2] + n_g_3_og[3] .* (s_xi) .^ n_exp[3]) #+C*log(T)
n_red_3_og  = exp.(n_g_3_og[1] .* (s_xi) .^ (1.8) + n_g_3_og[2] .* (s_xi) .^ (2.4) + n_g_3_og[3] .* (s_xi) .^ (2.8)) .- 1.0

# 4 param

n_g_4 = [-0.448046, 1.012681, -0.381869, 0.054674]



ln_n_red_4 = n_g_4[1] .* (s_xi) + n_g_4[2] .* (s_xi) .^ (1.5) + n_g_4[3] .* (s_xi) .^ (2) + n_g_4[4] .* (s_xi) .^ (2.5)


#n_g_3 = [0.281, -0.105, 0.0258]
#exponents = [1.7, 2.2, 2.5]

#n_g_3 = [0.2, -0.08, 0.045]
#exponents = [1.7, 2.3, 2.4]

#exp_dodecane = CSV.read("Training Data/Octane DETHERM.csv", DataFrame)
exp_dodecane = CSV.read("Training Data/Dodecane DETHERM.csv", DataFrame)
model = SAFTgammaMie(["Dodecane"])

function reduced_visc(model::EoSModel, P,T,visc)

    visc_CE = IB_CE(model,T)
    R = Rgas()

    s_res = entropy_res(model, P, T)
    s_red = -s_res ./ R

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B

    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A

    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A

    n_reduced = (visc - visc_CE) * ((s_red) .^ (2 / 3)) ./((ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T))
    return n_reduced
end

P = exp_dodecane[:,1]
T = exp_dodecane[:,2]
visc_exp = exp_dodecane[:,3]

eta_red = reduced_visc.(model,P,T,visc_exp)

y_axis = log.(eta_red .+ 1)
num_groups = model.groups.n_groups[1]
xi =   0.4085265*num_groups[1] + 0.0383325*num_groups[2]
x_axis = -entropy_res.(model,P,T)./(Rgas() .* xi)
#=
n_g_3 = [0.29136977, -0.1331025, 0.021]
exponents = [1.8, 2.3, 2.74]

ln_n_red_3 = n_g_3[1] .* (s_xi) .^ exponents[1] + n_g_3[2] .* (s_xi) .^ exponents[2] + n_g_3[3] .* (s_xi) .^ exponents[3]
=#

plot1 = plot(s_xi,ln_n_red_3_og,
grid = false,
lw =:3,
label = false,
xlabel = L"s^+ / \xi_\textrm{exp}",
ylabel = L"ln(\eta_\textrm{res}^+ +1)")
plot!(plot1,s_xi,ln_n_red_4,
lw =:3,
label = "4 param")
scatter!(plot1,x_axis,y_axis,
lw=:3,
marker =:diamond,
label = false,
xlims = (minimum(x_axis),maximum(x_axis)),
ylims = (minimum(y_axis),maximum(y_axis)))

plot2 = plot(s_xi,exp.(ln_n_red_3_og) .- 1.0,
grid = false,
lw =:3,
label = false,
xlabel = L"s^+ / \xi_\textrm{exp}",
ylabel = L"\eta_\textrm{res}^+")
scatter!(plot2,x_axis,eta_red,
lw=:3,
label = false,
marker =:diamond,
xlims = (minimum(x_axis),maximum(x_axis)),
ylims = (minimum(eta_red),maximum(eta_red)))
xlims!(2,4)
ylims!(0,3.5)

using CMAEvolutionStrategy
using Statistics, Random
using DataFrames
using CSV  # If loading from CSV; otherwise, you can hardcode x and y

# Function to load experimental data from CSV (adapt as needed)
function load_experimental_data(path::AbstractString)
    """
    Format experimental data from CSV
    """
    df = CSV.read(path, DataFrame)
    # Normalize column names to lowercase symbols
    rename!(df, Symbol.(lowercase.(String.(names(df)))))
    required = [:x_axis, :y_axis]  # Assuming columns are named x_axis and y_axis
    return df
end

# Objective Function
function make_global_objective(x_data::Vector, y_data::Vector)
    """
    Returns an objective function f(params) where:
    params = [n1, n2, n3, e1, e2, e3]
    """
    function objective(params)
        C ,n1, n2, n3, e1, e2, e3 = params
        e1 = 1.5
        e2 = 2.1
        e3 = 2.7
        n1 = 0.7235144301013621
        n2 = -0.2515309494126982
        n3 =  0.02919416238201568
        total_error = 0.0
        try
            y_pred = C .+ n1 .* (x_data .^ e1) .+ n2 .* (x_data .^ e2) .+ n3 .* (x_data .^ e3)
            if any(!isfinite, y_pred)
                total_error += 1e10
            else
                total_error += sum(((y_data .- y_pred)./y_data).^2) / length(x_data)
            end
        catch err
            @warn "Invalid point encountered during optimization" params = params error = err
            total_error += 1e10
        end
        return isfinite(total_error) ? total_error : 1e10
    end
    return objective
end

# CMA-ES Optimization
function estimate_parameters_CMA!(x_data::Vector, y_data::Vector;
    lower = [-10.0],
    upper = [10.0],
    seed = 42, σ0 = 0.1, max_iters = 5000)
    Random.seed!(seed)
    obj = make_global_objective(x_data, y_data)
    # Initial guess: use provided for exponents, midpoints for n1,n2,n3 if not specified
    x0 = [0.0]  # Assuming initial n1,n2,n3=0.0 (midpoint of example bounds)
    println("Starting CMA-ES optimization (n1, n2, n3, e1, e2, e3) — seed=$seed")
    println("Initial guess: ", x0)
    iter_counter = Ref(0)
    result = minimize(
        obj,
        x0,
        σ0;
        lower = lower,
        upper = upper,
        seed = seed,
        verbosity = 2,
        maxiter = max_iters,
        ftol = 1e-9,
        callback = (opt, x, fx, ranks) -> begin
            iter_counter[] += 1
            if iter_counter[] % 20 == 0
                println("Iter $(iter_counter[]): fmin=$(minimum(fx)) best=$(xbest(opt))")
            end
        end
    )
    println("\nCMA-ES optimization complete.")
    println("Best parameters found:")
    println(xbest(result))
    println("Objective value = ", fbest(result))
    return result
end

# === Example Usage ===
# If data is in CSV:
# data_paths = ["your_data.csv"]
# datasets = [load_experimental_data(p) for p in data_paths]
# x_data = datasets[1].x_axis
# y_data = datasets[1].y_axis

# Or hardcode your data here (replace with actual values):
x_data = x_axis
y_data = y_axis

# Run optimization
res = estimate_parameters_CMA!(
    x_data,
    y_data;
    lower = [-5.0],
    upper = [5.0],
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)
println("\nBest solution (CMA-ES): ", xbest(res))
println("Best objective value: ", fbest(res))
#0.0031868