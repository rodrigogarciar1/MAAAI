using Pkg
Pkg.activate("../environment")
Pkg.instantiate()

using Statistics

# ===============================
# Pearson SelectKBest Filter
# ===============================
# Definir la estructura del modelo
mutable struct PearsonFilter <: MLJModelInterface.Supervised
    k::Int
end

# Constructor con argumentos por nombre (estilo MLJ)
PearsonFilter(; k::Int=2) = PearsonFilter(k)

# Se hace la importación de las funciones necesarias para poder sobrecargarlas
import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype

# Definir la función fit
function fit(model::PearsonFilter, verbosity::Int, X, y)
    # Convertir X a matriz y asegurarse que es Float64
    Xmat = Float64.(MLJBase.matrix(X))
    
    # Convertir y a vector numérico si es categórico
    y_numeric = y isa CategoricalVector ? Float64.(levelcode.(y)) : Float64.(y)
    
    # Calcular correlación de Pearson para cada característica
    n_features = size(Xmat, 2)
    correlations = zeros(Float64, n_features)
    
    for j in 1:n_features
        feature = Xmat[:, j]
        
        # Verificar que la característica tiene varianza
        if std(feature) == 0.0
            correlations[j] = 0.0
            continue
        end
        
        try
            # Calcular correlación de Pearson
            correlations[j] = abs(cor(feature, y_numeric))
        catch e
            # Si hay algún error en el cálculo, asignar 0
            correlations[j] = 0.0
        end
    end
    
    # Seleccionar top k features con mayor correlación absoluta
    k_actual = min(model.k, n_features)
    idxs = sortperm(correlations, rev=true)[1:k_actual]
    
    # Guardar los nombres de las columnas originales
    feature_names = collect(Tables.columnnames(X))
    selected_names = [feature_names[i] for i in idxs]
    
    # IMPORTANTE: fitresult debe contener la info necesaria para transform
    fitresult = (idxs=idxs, selected_names=selected_names)
    cache = nothing
    report = (correlations=correlations, idxs=idxs, selected_features=selected_names)
    
    return fitresult, cache, report
end

# Definir la función transform
function transform(model::PearsonFilter, fitresult, X)
    # Convertir X a matriz
    Xmat = MLJBase.matrix(X)
    
    # Seleccionar columnas usando fitresult (no cache)
    X_selected = Xmat[:, fitresult.idxs]
    
    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end

# Definir los tipos de entrada y salida
input_scitype(::Type{<:PearsonFilter}) = Table(Continuous)
target_scitype(::Type{<:PearsonFilter}) = AbstractVector{<:Finite}
output_scitype(::Type{<:PearsonFilter}) = Table(Continuous)