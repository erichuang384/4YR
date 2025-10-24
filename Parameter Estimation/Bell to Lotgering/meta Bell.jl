using Metaheuristics
using CSV, DataFrames, Random

# Read the data
fit_data = CSV.read("Parameter Estimation/Bell to Lotgering/Bell iv coeffs.csv", DataFrame)

A_data = fit_data[:, 2]
B_data = fit_data[:, 3]
C_data = fit_data[:, 4]
n_ch2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
n_ch3 = fill(2, length(A_data))

# --- Define Objective Functions for A, B, and C ---
function sse_loss_AA(params, A_data, n_ch2, n_ch3)
    A_ch3, A_ch2 = params
    #A_model = n_ch3 .* A_ch3 .+ n_ch2 .* A_ch2
    x_ch3 = n_ch3./(n_ch3 .+ n_ch2)
    x_ch2 = n_ch2./(n_ch3 .+ n_ch2)
    A_model = n_ch3.^1 .* A_ch3 .^ 1 .+ n_ch2.^1 .* A_ch2 .^ 1 #+  2 .* x_ch3 .* x_ch2 .* (A_ch3 .* A_ch2)
    return sum(abs.((A_data .- A_model)./A_data))
end

function sse_loss_BB(params, B_data, n_ch2, n_ch3)
    B_ch3, B_ch2, gamma = params
    shapef = [0.57255, 0.22932]
    sigmas = [4.0772, 4.8801]
    x_ch3 = n_ch3./(n_ch3 .+ n_ch2)
    x_ch2 = n_ch2./(n_ch3 .+ n_ch2)

    V_tot = n_ch3 .* shapef[1] .* sigmas[1] .^ 3 + n_ch2 .* shapef[2] .* sigmas[2] .^ 3
    B_model = (x_ch3 .^1 .* B_ch3 .^ 1 .+ x_ch2.^1 .* B_ch2 .^1)./(V_tot .^ gamma)
    #gamma = 0.45
    #B_model = (n_ch3 .* B_ch3 .+ n_ch2 .* B_ch2)./(V_tot .^ gamma)

    #B_ch3, B_ch2 = params
    #B_model = n_ch3 .* B_ch3 .+ n_ch2 .* B_ch2
    return sum(abs.((B_data .- B_model)./B_data))
end

B_optimum_result = optimize_parameter_with_metaheuristics(sse_loss_BB, lower_bound, upper_bound, B_data, n_ch2, n_ch3, seed=42, max_iters=10000)


function sse_loss_C1(params, C_data, n_ch2, n_ch3)
    C_ch3, C_ch2, gamma = params

    x_ch3 = n_ch3./(n_ch3 .+ n_ch2)
    x_ch2 = n_ch2./(n_ch3 .+ n_ch2)
    x = [x_ch3, x_ch2]

    #C_model = cbrt.(x_ch3.^2 .* C_ch3 .^ 3 .+ x_ch2.^2 .* C_ch2 .^ 3  .+ 2 .* x_ch3 .* x_ch2 .* (C_ch3 .* C_ch2).^(3/2)) #+  1 .* x_ch3 .* x_ch2 .* sqrt.(C_ch3 .* C_ch2)

    shapef = [0.57255, 0.22932]
    sigmas = [4.0772, 4.8801]

    V_tot = n_ch3 .* shapef[1] .* sigmas[1] .^ 3 + n_ch2 .* shapef[2] .* sigmas[2] .^ 3
    #gamma = 1.4
    #C_model = (x_ch3 .^1 .* C_ch3 .^ 1 .+ x_ch2.^1 .* C_ch2 .^1 .+ 0 .*x_ch2.*x_ch3 .*(C_ch3.*C_ch2).^0.5)./(V_tot .^ gamma)
    C_model = (x_ch3 .^1 .* C_ch3 .^ 1 .+ x_ch2.^1 .* C_ch2 .^1 )./(V_tot .^ gamma)
    #C_model = (n_ch3./((n_ch3+n_ch2) .* C_ch3) .+ n_ch2./((n_ch3+n_ch2) .* C_ch2)) .^ (-1)

    return sum(abs.((C_data .- C_model)./C_data))
end


# --- Define the Optimization Wrapper ---
function optimize_parameter_with_metaheuristics(loss_function, lower_bound, upper_bound, data, n_ch2, n_ch3; seed=1234, max_iters=2000)
    rng = MersenneTwister(seed)
    Random.seed!(rng)

    bounds = [lower_bound upper_bound]'

    method = DE()
    method.options = Options(iterations = max_iters, f_calls_limit = 1_000_000, store_convergence = true, seed = seed)

    logger = function (status)
        if isdefined(status, :iteration)
            if status.iteration % 50 == 0 || status.iteration == 1
                println("iter=$(status.iteration) f_calls=$(status.f_calls) best_sol=$(status.best_sol)")
            end
        end
    end

    println("Starting optimization...")
    state = Metaheuristics.optimize(
        p -> loss_function(p, data, n_ch2, n_ch3), bounds, method; logger = logger)

    result = Metaheuristics.get_result(method)
    return result
end
# --- Run Optimization for Each Parameter ---
# Initial guesses for the parameters
initial_guess = [0.0, 0.0]

# Bounds for each parameter
lower_bound = [-100.0, -100.0, -100]  # Set lower bounds for parameters
upper_bound = [100.0, 1000.0, 100]    # Set upper bounds for parameters

# Optimize A
A_optimum_result = optimize_parameter_with_metaheuristics(sse_loss_AA, lower_bound, upper_bound, A_data, n_ch2, n_ch3, seed=42, max_iters=20000)
println("A optimization result:", A_optimum_result.best_sol)

# Optimize B
B_optimum_result = optimize_parameter_with_metaheuristics(sse_loss_BB, lower_bound, upper_bound, B_data, n_ch2, n_ch3, seed=42, max_iters=1000)
println("B optimization result:", B_optimum_result.best_sol)

# Optimize C
C_optimum_result = optimize_parameter_with_metaheuristics(sse_loss_C1, lower_bound, upper_bound, C_data, n_ch2, n_ch3, seed=42, max_iters=30000)
println("C optimization result:", C_optimum_result.best_sol)