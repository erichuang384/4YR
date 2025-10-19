using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings, StaticArrays
include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))

# ---------------------------------------------------------------------
# === VISCOSITY FUNCTION (your original) ==============================
function IB_viscosity(model::EoSModel, P, T, z = StaticArrays.SA[1.0])
    n_g = [-0.448046, 1.012681, -0.381869, 0.054674]
    ξ_pure = zeros(length(z))

    for j ∈ 1:length(z)
        ξ_i = ["CH3" 0.484458; "CH2" 0.047793]
        ξ = 0.0
        groups = model.groups.groups[j]
        num_groups = model.groups.n_groups[j]
        for i in 1:length(groups)
            value = ξ_i[ξ_i[:, 1].==groups[i], 2][1]
            ξ += value * num_groups[i]
        end
        ξ_pure[j] = ξ
    end

    ξ_mix = sum(z .* ξ_pure)
    R = Rgas()
    s_res = entropy_res(model, P, T, z)
    s_red = -s_res ./ R

    n_reduced = exp(n_g[1] .* (s_red ./ ξ_mix) + n_g[2] .* (s_red ./ ξ_mix).^1.5 +
                    n_g[3] .* (s_red ./ ξ_mix).^2 + n_g[4] .* (s_red ./ ξ_mix).^2.5) - 1

    N_A = Clapeyron.N_A
    k_B = Clapeyron.k_B
    ρ_molar = molar_density(model, P, T, z)
    ρ_N = ρ_molar .* N_A
    Mw = Clapeyron.molecular_weight(model, z)
    m = Mw / N_A

    n_res = (n_reduced .* (ρ_N .^ (2 / 3)) .* sqrt.(m .* k_B .* T)) ./ (s_red .^ (2 / 3))

    viscosity = (length(z) == 1) ? IB_CE(model, T) + n_res : IB_CE_mix(model, T, z) + n_res
    return viscosity
end

# ---------------------------------------------------------------------
# === LOAD DATA ========================================================
function load_binary_data(path::AbstractString)
    df = CSV.read(path, DataFrame)
    #rename!(df, lowercase.(names(df))) # ensure lowercase columns
    df.z = [SVector(x, 1 - x) for x in df[:, 3]]
    return df
end

# ---------------------------------------------------------------------
# === COMPUTE MODEL VALUES ============================================
function compute_viscosity_curve(model::EoSModel, p::Real, T::Real; n_points::Int=300)
    x = LinRange(0, 1, n_points)
    visc = [IB_viscosity(model, p, T, @SVector [xi, 1 - xi]) for xi in x]
    return x, visc
end


default(
    linewidth = 2,
    framestyle = :box,
    grid = false,
    minorgrid = false,
    size = (700, 500),
    background_color = :white,
)

function plot_binary_viscosities(data_map::Dict{String,DataFrame}, models)
    p = 101325
    Tvals = [293.15, 298.15]

    # Different marker shapes for each mixture
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :hexagon]
    names = collect(keys(data_map))

    plots = Dict{Float64, Any}()

    for T in Tvals
        plt = plot(
            xlabel = L"x_1",
            ylabel = L"\eta \, / \, \mathrm{Pa \, s}",
            title = "Viscosity at $(T) K",
            legend = false
        )

        for (i, name) in enumerate(names)
            df = data_map[name]
            idx = findall(df[:, 2] .≈ T)
            isempty(idx) && continue

            x_exp = df[idx, 3]
            visc_exp = df[idx, 4]
            model = models[name]

            # Experimental data
            scatter!(
                plt, x_exp, visc_exp,
                marker = markers[mod1(i, length(markers))],
                color = :black,
                markersize = 4
            )

            # Model prediction
            x_line, visc_line = compute_viscosity_curve(model, p, T)
            plot!(plt, x_line, visc_line, color = :blue, linewidth = 2)
        end

        plots[T] = plt
    end

    return plots
end
# ---------------------------------------------------------------------
# === MAIN SCRIPT =====================================================
function main()
    data_dir = "Experimental Data/Binary Liquid n-alkane"
    mixture_files = [
        "decane_pentadecane.csv",
        "decane_tridecane.csv",
        "octane_pentadecane.csv",
        "octane_tridecane.csv",
        "octane_undecane.csv",
        "tridecane_pentadecane.csv",
        "undecane_pentadecane.csv",
        "undecane_tridecane.csv",
    ]

    # load all datasets
    data_map = Dict{String,DataFrame}()
    for file in mixture_files
        key = replace(file, ".csv" => "")
        data_map[key] = load_binary_data(joinpath(data_dir, file))
    end

    # define models for each binary
    models = Dict(
        "decane_pentadecane"   => SAFTgammaMie(["Decane", "Pentadecane"]),
        "decane_tridecane"     => SAFTgammaMie(["Decane", "Tridecane"]),
        "octane_pentadecane"   => SAFTgammaMie(["Octane", "Pentadecane"]),
        "octane_tridecane"     => SAFTgammaMie(["Octane", "Tridecane"]),
        "octane_undecane"      => SAFTgammaMie(["Octane", "Undecane"]),
        "tridecane_pentadecane"=> SAFTgammaMie(["Tridecane", "Pentadecane"]),
        "undecane_pentadecane" => SAFTgammaMie(["Undecane", "Pentadecane"]),
        "undecane_tridecane"   => SAFTgammaMie(["Undecane", "Tridecane"]),
    )

    # generate plots
    plots = plot_binary_viscosities(data_map, models)

    # show or save
    display(plots[293.15])
    display(plots[298.15])
    savefig(plots[293.15], "viscosity_293K.png")
    savefig(plots[298.15], "viscosity_298K.png")
end

main()
