using Pkg
Pkg.activate("../environment")
Pkg.instantiate()

using InformationMeasures

# ===============================
# MutualInformation SelectKBest Filter
# ===============================
# Definir la estructura del modelo
mutable struct MutualInformation <: MLJModelInterface.Supervised
    k::Int
end

# Constructor con argumentos por nombre (estilo MLJ)
MutualInformation(; k::Int=2) = MutualInformation(k)

# Se hace la importación de las funciones necesarias para poder sobrecargarlas
import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype

# Definir la función fit
function fit(model::MutualInformation, verbosity::Int, X, y)
    # Convertir X a matriz y asegurarse que es Float64
    Xmat = Float64.(MLJBase.matrix(X))
    
    # Convertir y a vector numérico si es categórico
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)
    
    # Calcular F-statistics usando HypothesisTests
    n_features = size(Xmat, 2)
    
    fstats = get_mutual_information.(eachcol(Xmat), Ref(y_numeric))
    
    # Seleccionar top k features con mayor F-statistic
    k_actual = min(model.k, n_features)
    idxs = sortperm(fstats, rev=true)[1:k_actual]
    
    # Guardar los nombres de las columnas originales
    feature_names = collect(Tables.columnnames(X))
    selected_names = [feature_names[i] for i in idxs]
    
    # IMPORTANTE: fitresult debe contener la info necesaria para transform
    fitresult = (idxs=idxs, selected_names=selected_names)
    cache = nothing
    report = (fstats=fstats, idxs=idxs, selected_features=selected_names)
    
    return fitresult, cache, report
end

# Definir la función transform
function transform(model::MutualInformation, fitresult, X)
    # Convertir X a matriz
    Xmat = MLJBase.matrix(X)
    
    # Seleccionar columnas usando fitresult
    X_selected = Xmat[:, fitresult.idxs]
    
    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end     

# Definir los tipos de entrada y salida
input_scitype(::Type{<:MutualInformation}) = Table(Continuous)
target_scitype(::Type{<:MutualInformation}) = AbstractVector{<:Finite}
output_scitype(::Type{<:MutualInformation}) = Table(Continuous)