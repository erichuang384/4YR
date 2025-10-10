using Clapeyron, Plots, LinearAlgebra, CSV, DataFrames, LaTeXStrings

include("all_functions.jl")

model_hexane=SAFTgammaMie(["hexane"])
#n_hexane = reduced_viscosity(model_hexane,1e5,320)   

N=100
T_range=LinRange(200,350,N)
P=1e5

function dilute_gas_viscosity(model::EoSModel,T)
    """
    Chapman-Enskog Theory
    """
    N_A = 6.0221408e23
    k_B = 1.380649e-23
    critical = crit_pure(model)
    m = molar_density(model,critical[2],critical[1])
    σ = 0.809 * (m*100^(-3))^(-1/3) * 10^(-10)

    Mw = Clapeyron.molecular_weight(model) # in kg/mol
    m_gc = sum(model.groups.n_groups[1])
    Ω = Ω⃰(model,T)
    visc = 5/16*sqrt(Mw*k_B*T/(N_A*pi))/(σ^2*Ω)
    return visc
end

function Ω⃰(model::EoSModel,T)
    """
    Reduced Collision Integral
    For two component, need to work on getting epsilon in general
    """

    critical = crit_pure(model)
    #m = molar_density(model,critical[2],critical[1])
    ϵ = critical[1]/1.2593
    T⃰ = T / ϵ
    if !(0.3 < T⃰ < 100)
        print("Unviable T for correlation")
    end
    #think that k_B is already in epsilon parameter according to sample calc
    Ω = 1.16145*(T⃰)^(-0.14874)+0.52487*exp(-0.77320*(T⃰))+2.16178*exp(-2.43787*(T⃰))
    return Ω
end


function reduced_viscosity(model::EoSModel,P,T)

    #n_α= model.groups.n_groups[1] 
    #S=model.params.shapefactor
    #σ = diag(model.params.sigma.values)
    #m_alpha_3 = [34.16955e-30, 24.33981e-30] # for CH3, CH2
    #A_α=[−8.6878e-3,−0.9194e-3];
    #B_α=[-1.7951e-10,−1.3316e-10]  #paramters from Lotgering
    #C_α=[−12.2359e-2,−4.2657e-2]

    #γ=0.45 #Constant for n-alkanes

    #V=sum(n_α.*m_alpha_3)
    A= -1.2035
    B=  -2.5958
    C =  -0.4816
    D = -0.0865
    
    s_res=entropy_res(model,P,T)
    #m_gc = sum(model.groups.n_groups[1])
    k_B = 1.380649e-23
    R = 8.31446261815324 #Using R instead of kB
    scaling_factor = PCPSAFT(model.groups.components).params.segment.values[1]

    z=(s_res./(R.*scaling_factor))  # molar entropy

    n_reduced=exp(A+B.*z+C.*z^2+D.*z^3)
    return n_reduced
end

#n_reduced=zeros(N)
#n_CE=zeros(N)
viscosity=zeros(N)
for i in 1:N
    vicosity_reduced= reduced_viscosity(model_hexane,P,T_range[i])
    viscosity_CE = dilute_gas_viscosity(model_hexane,T_range[i]) 
    viscosity[i]=vicosity_reduced.*viscosity_CE
end

# Plot viscosity vs. temperature
plot(
    T_range, viscosity,
    xlabel = L"T\ \mathrm{(K)}",
    ylabel = L"\eta\ \mathrm{(Pa\,s)}",
    title  = "Viscosity of Hexane at 1 bar",
    lw = 2,
    grid = false,
    legend = false,
)

df = CSV.read("Hexane 20bar viscosity.csv", DataFrame)


T = df[:, 1]
n_experimental = df[:, 3] 

# Add experimental data as points
plot!(T, n_experimental;
      seriestype = :scatter,    # points instead of lines
      label = "Experimental",    # circle markers, size 6
)  