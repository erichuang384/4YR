using Optim
using Printf
using CSV, DataFrames

fit_data = CSV.read("Parameter Estimation/Bell to Lotgering/quartic_fits.csv", DataFrame)

# --- 1. Define Data ---
A_data = fit_data[:, 2]  # Adjust column index for A if needed
B_data = fit_data[:,3]
C_data = fit_data[:, 4]  # Adjust column index for C if needed

Mw = [15.035, 14.027]

#shapef = [0.57255, 0.22932]
#sigmas = [4.0772, 4.8801]

n_ch2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
n_ch3 = fill(2, length(A_data))

# --- 2. Define Objective Functions ---
function sse_loss_A(p)
    A_ch3, A_ch2 = p
    A_model = n_ch3 .* A_ch3 .+ n_ch2 .* A_ch2
    return sum((A_data .- A_model).^2)
end

function sse_loss_B(p)
    B_ch3, B_ch2,D = p
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
    D_ch3, D_ch2 = p
    #D_2 = p
    Mw = [15.035, 14.027]
    Mw_tot = n_ch3 .* Mw[1] .+ n_ch2 .* Mw[2]
    #D_model = (D_2./Mw_tot .+ D_1).^(-2)
    D_model = n_ch3 .* D_ch3 .+ n_ch2 .* D_ch2
    return sum((D_data .- D_model).^2)
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

# --- 3. Run Optimization for Both Parameters ---

function optimize_parameter(param_name, loss_function, initial_guess)
    println("\n" * "="^50)
    println("Optimizing parameter $param_name...")
    
    result = Optim.optimize(loss_function, initial_guess, LBFGS())
    
    if Optim.converged(result)
        optimum_params = Optim.minimizer(result)
        final_sse = loss_function(optimum_params)
        
        #abs_relative_error = mean(abs.((data .- model_pred) ./ data))
        
        println("Optimization Successful!")
        @printf "Optimum %s_CH3: %.7f\n" param_name optimum_params[1]
        @printf "Optimum %s_CH2: %.7f\n" param_name optimum_params[2]
        @printf "Final SSE: %.7f\n" final_sse
        #@printf "Mean Absolute Relative Error: %.4f%%\n" (abs_relative_error * 100)sse
        
        return optimum_params
    else
        println("Optimization for $param_name did not converge.")
        return nothing
    end
end



# --- 4. Execute Optimizations ---
initial_guess = [00.0, 0.0, -0.45]

# Optimize A
A_optimum = optimize_parameter("A", sse_loss_A, initial_guess)

# Optimize B
B_optimum = optimize_parameter("B", sse_loss_B, initial_guess)

# Optimize C  
C_optimum = optimize_parameter_C("C", sse_loss_C,nothing ,initial_guess)
# Optimize D
D_optimum = optimize_parameter("D", sse_loss_D, initial_guess)

