using Clapeyron, CSV, DataFrames

include(joinpath(dirname(@__FILE__), "..", "bell_functions.jl"))

# AAD for training set

models = [
    SAFTgammaMie(["Octane"]),
    SAFTgammaMie(["Decane"]),
    SAFTgammaMie([("Dodecane",["CH3"=>2,"CH2"=>10])]),
    SAFTgammaMie([("Tetradecane",["CH3"=>2,"CH2"=>12])]),
    SAFTgammaMie([("Hexadecane",["CH3"=>2,"CH2"=>14])])
]

data_paths = [
    "Training DATA/Octane DETHERM.csv",
    "Training DATA/Decane DETHERM.csv",
    "Training DATA/Dodecane DETHERM.csv",
    "Training DATA/Tetradecane DETHERM.csv",
    "Training DATA/Hexadecane DETHERM.csv"
]

datasets = [load_experimental_data(p) for p in data_paths]

# Viscosities
