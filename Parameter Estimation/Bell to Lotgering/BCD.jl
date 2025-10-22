using Metaheuristics
using Printf
using CSV, DataFrames

# Read the data (ensure path is correct)
fit_data = CSV.read("Parameter Estimation/Bell to Lotgering/quartic_fits.csv", DataFrame)

# --- 1. Define Data ---
A_data = fit_data[:, 2]  # Adjust column index for A if needed
B_data = fit_data[:,3]
C_data = fit_data[:, 4]  # Adjust column index for C if needed
D_data = fit_data[:,5]
Mw = [15.035, 14.027]

n_ch2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
n_ch3 = fill(2, length(A_data))

# --- 2. Define Objective Functions ---
function sse_loss_A(p)
    A_ch3, A_ch2 = p
    A_model = n_ch3 .* A_ch3 .+ n_ch2 .* A_ch2
    return sum((A_data .- A_model).^2)
end

function sse_loss_B(p)
    B_ch3, B_ch2, D = p
    shapef = [0.57255, 0.22932]
    sigmas = [4.0772, 4.8801]

    V_tot = n_ch3 .* shapef[1] .* sigmas[1] .^ 3 + n_ch2 .* shapef[2] .* sigmas[2] .^ 3
    gamma = D
    B_model = (n_ch3 .* B_ch3 .+ n_ch2 .* B_ch2)./(V_tot .^ gamma)
    return sum((B_data .- B_model).^2)
end

function sse_loss_C(p)
    C_ch3, C_ch2 = p
    C_model = n_ch3 .* C_ch3 .+ n_ch2 .* C_ch2 .+ (n_ch3 .+ n_ch2) .* C_ch3 .* C_ch2
    return sum((C_data .- C_model).^2)
end

function sse_loss_D(p)
    D_ch3, D_ch2, D = p
    shapef = [0.57255, 0.22932]
    sigmas = [4.0772, 4.8801]

    V_tot = n_ch3 .* shapef[1] .* sigmas[1] .^ 3 + n_ch2 .* shapef[2] .* sigmas[2] .^ 3
    gamma = D
    D_model = (n_ch3 .* D_ch3 .+ n_ch2 .* D_ch2)./(V_tot .^ gamma)
    return sum((D_data .- D_model).^2)
end

# --- 3. Set Up Optimization with Metaheuristics ---
function optimize_parameter(param_name, loss_function, bounds)
    println("\n" * "="^50)
    println("Optimizing parameter $param_name...")
    
    # Set up global optimization using a Genetic Algorithm (you can replace this with other algorithms like SimulatedAnnealing or DifferentialEvolution)
    optimizer = DE(loss_function, bounds)
    
    result = optimize(optimizer)
    
    # Output the result
    if result.fitness < 1e-6  # Adjust threshold as needed
        optimum_params = result.best_position
        final_sse = result.fitness
        
        println("Optimization Successful!")
        @printf "Optimum %s_CH3: %.7f\n" param_name optimum_params[1]
        @printf "Optimum %s_CH2: %.7f\n" param_name optimum_params[2]
        @printf "Final SSE: %.7f\n" final_sse
        return optimum_params
    else
        println("Optimization for $param_name did not converge.")
        return nothing
    end
end

# --- 4. Set Bounds for the Optimization ---
# These bounds are dependent on the parameters you are optimizing
bounds_A = [(-10.0, 10.0), (-10.0, 10.0)]  # Example bounds for A parameters
bounds_B = [(-10.0, 10.0), (-10.0, 10.0), (-5.0, 5.0)]  # Example bounds for B
bounds_C = [(-10.0, 10.0), (-10.0, 10.0)]  # Example bounds for C
bounds_D = [(-10.0, 10.0), (-10.0, 10.0), (-5.0, 5.0)]  # Example bounds for D

# --- 5. Execute Optimizations ---
initial_guess = [0.0, 0.0]  # Example initial guess for all parameters

# Optimize A
A_optimum = optimize_parameter("A", sse_loss_A, bounds_A)

# Optimize B
B_optimum = optimize_parameter("B", sse_loss_B, bounds_B)

# Optimize C
C_optimum = optimize_parameter("C", sse_loss_C, bounds_C)

# Optimize D
D_optimum = optimize_parameter("D", sse_loss_D, bounds_D)
