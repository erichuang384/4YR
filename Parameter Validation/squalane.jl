using Clapeyron, CSV, DataFrames, Plots, LaTeXStrings
cd("/Users/erich/OneDrive/Documents/Y4/4YR")
model = SAFTgammaMie(["Squalane"])

exp_data = CSV.read("Atmospheric Data/squalane density.csv",DataFrame)


T = [278.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15]
den_array = zeros(length(exp_data[:,1]))

den_array = mass_density.(model,exp_data[:,2],exp_data[:,1])

AAD = sum((abs.(exp_data[:,3] .- den_array))./exp_data[:,3])/length(den_array)
