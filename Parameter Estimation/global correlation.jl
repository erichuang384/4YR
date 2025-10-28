using CMAEvolutionStrategy
using Statistics, Random
using DataFrames, CSV, Clapeyron
using Plots, LaTeXStrings

# Function to load experimental data from CSV
function load_experimental_data(path::AbstractString)
    """
    Format experimental data from CSV
    """
    df = CSV.read(path, DataFrame)
    # Normalize column names to lowercase symbols
    rename!(df, Symbol.(lowercase.(String.(names(df)))))
    required = [:p, :t, :viscosity]
    return df
end

# Function to compute reduced viscosity
function reduced_visc(model::EoSModel, P, T, visc)
    visc_CE = IB_CE(model, T)
    R = Rgas()
    s_res = entropy_res(model, P, T)
    s_red = -s_res ./ R
    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T)
    ρ_N = ρ_molar .* N_A
    Mw = Clapeyron.molecular_weight(model)
    m = Mw / N_A
    n_reduced = (visc .- visc_CE) .* ((s_red) .^ (2 / 3)) ./ ((ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T))
    return n_reduced
end

# Objective Function
function make_global_objective(models::Vector, datasets::Vector{DataFrame})
    """
    Returns an objective function f(params) where:
    params = [C, n1, n2, n3, e1, e2, e3]
    """
    function objective(params)
        C, n1, n2, n3 = params
        e1 = 1.4
        e2 = 2.25
        e3 = 2.6
        total_error = 0.0
        
        for (model, data) in zip(models, datasets)
            try
                P = data.p
                T = data.t
                visc_exp = data.viscosity
                num_groups = model.groups.n_groups[j] #n-element Vector{Int}
                xi = 0.4085265 * num_groups[1] + 0.0383325 * num_groups[2]  # As provided in original code
                eta_red = reduced_visc.(model, P, T, visc_exp)
                y_data = log.(eta_red .+ 1)
                x_data = -entropy_res.(model, P, T) ./ (Rgas() .* xi)
                y_pred = C .+ n1 .* (x_data .^ e1) .+ n2 .* (x_data .^ e2) .+ n3 .* (x_data .^ e3)
                if any(!isfinite, y_pred)
                    total_error += 1e10
                    continue
                end
                total_error += sum(((y_data .- y_pred) ./ y_data) .^ 2) / length(x_data)
            catch err
                @warn "Invalid point encountered during optimization" params = params error = err
                total_error += 1e10
            end
        end
        return isfinite(total_error) ? total_error : 1e10
    end
    return objective
end

# CMA-ES Optimization
function estimate_parameters_CMA!(models::Vector, datasets::Vector{DataFrame};
    lower = [0.0, -1.0, 0.0, -10.0],
    upper = [1.0, 0.0, 1.0, 10.0],
    seed = 42, σ0 = 0.1, max_iters = 10000)
    Random.seed!(seed)
    obj = make_global_objective(models, datasets)
    # Initial guess: C=0, n1,n2,n3 from provided means, e1,e2,e3 as specified
    x0 = [0.0, 0.7235144301013621, -0.2515309494126982, 0.02919416238201568]
    println("Starting CMA-ES optimization (C, n1, n2, n3, e1, e2, e3) — seed=$seed")
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
    println("Best parameters found: [C, n1, n2, n3, e1, e2, e3] = ", xbest(result))
    println("Objective value = ", fbest(result))
    return result
end

# Plotting Function
function plot_results(models, datasets, params)
    C, n1, n2, n3 = params
    e1 = 1.4
    e2 = 2.25
    e3 = 2.6
    xi = 0.4085265 * num_groups[1] + 0.0383325 * num_groups[2] 
    L = 1000
    s_xi = LinRange(0, 20, L)
    ln_n_red = C .+ n1 .* (s_xi .^ e1) .+ n2 .* (s_xi .^ e2) .+ n3 .* (s_xi .^ e3)
    n_red = exp.(ln_n_red) .- 1.0

    plot1 = plot(s_xi, ln_n_red,
        grid = false,
        lw = 3,
        label = "Fitted Model",
        xlabel = L"s^+ / \xi_\textrm{exp}",
        ylabel = L"ln(\eta_\textrm{res}^+ + 1)")
    
    plot2 = plot(s_xi, n_red,
        grid = false,
        lw = 3,
        label = "Fitted Model",
        xlabel = L"s^+ / \xi_\textrm{exp}",
        ylabel = L"\eta_\textrm{res}^+")

    for (model, data) in zip(models, datasets)
        P = data.p
        T = data.t
        visc_exp = data.viscosity
        eta_red = reduced_visc.(model, P, T, visc_exp)
        y_axis = log.(eta_red .+ 1)
        x_axis = -entropy_res.(model, P, T) ./ (Rgas() .* xi)
        
        scatter!(plot1, x_axis, y_axis,
            lw = 3,
            label = "$(model.components[1]) Data",
            xlims = (minimum(x_axis), maximum(x_axis)),
            ylims = (minimum(y_axis), maximum(y_axis)))
        
        scatter!(plot2, x_axis, eta_red,
            lw = 3,
            label = "$(model.components[1]) Data",
            xlims = (2, 5),
            ylims = (0, 8))
    end

    return plot1, plot2
end

# Models and Data
models = [
    SAFTgammaMie(["Butane"]),
    SAFTgammaMie(["Hexane"]),
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie(["Dodecane"]),
    SAFTgammaMie(["Tetradecane"]),
    SAFTgammaMie(["Hexadecane"])
]

data_paths = [
    "Training DATA/Butane DETHERM.csv",
    "Training DATA/Hexane DETHERM.csv",
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Training DATA/Tetradecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv"
]

datasets = [load_experimental_data(p) for p in data_paths]

# Run optimization

res = estimate_parameters_CMA!(
    models,
    datasets;
    lower = [-3.0, 0.0, -1.0, 0.0],
    upper = [1.0, 1.0, 0.0, 1.0],
    seed = 42,
    σ0 = 0.1,
    max_iters = 10000
)

# Plot results
plot1, plot2 = plot_results(models, datasets, xbest(res))
display(plot1)
display(plot2)