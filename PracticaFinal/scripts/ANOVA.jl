using Pkg
Pkg.activate("../environment")
Pkg.instantiate()

using HypothesisTests

# ===============================
# ANOVA SelectKBest Filter
# ===============================
# Definir la estructura del modelo
mutable struct ANOVAFilter <: MLJModelInterface.Supervised
    k::Int
end

# Constructor con argumentos por nombre (estilo MLJ)
ANOVAFilter(; k::Int=2) = ANOVAFilter(k)

# Se hace la importación de las funciones necesarias para poder sobrecargarlas
import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype

# Definir la función fit
function fit(model::ANOVAFilter, verbosity::Int, X, y)
    # Convertir X a matriz y asegurarse que es Float64
    Xmat = Float64.(MLJBase.matrix(X))
    
    # Convertir y a vector numérico si es categórico
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)
    
    # Calcular F-statistics usando HypothesisTests
    n_features = size(Xmat, 2)
    fstats = zeros(Float64, n_features)
    
    for j in 1:n_features
        feature = Xmat[:, j]
        
        # Agrupar datos por clase
        classes = unique(y_numeric)
        groups = [feature[y_numeric .== c] for c in classes]
        
        # Filtrar grupos vacíos
        groups = filter(g -> length(g) > 0, groups)
        
        if length(groups) < 2
            fstats[j] = 0.0
            continue
        end
        
        try
            # Crear test ANOVA
            test = OneWayANOVATest(groups...)
            
            # Calcular F-statistic manualmente desde los campos disponibles
            MSt = test.SStᵢ / test.DFt  # Mean square treatment (between)
            MSe = test.SSeᵢ / test.DFe  # Mean square error (within)
            
            fstats[j] = MSt / MSe
        catch e
            # Si hay algún error en el cálculo, asignar 0
            fstats[j] = 0.0
        end
    end
    
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
function transform(model::ANOVAFilter, fitresult, X)
    # Convertir X a matriz
    Xmat = MLJBase.matrix(X)
    
    # Seleccionar columnas usando fitresult (no cache)
    X_selected = Xmat[:, fitresult.idxs]
    
    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end     

# Definir los tipos de entrada y salida
input_scitype(::Type{<:ANOVAFilter}) = Table(Continuous)
target_scitype(::Type{<:ANOVAFilter}) = AbstractVector{<:Finite}
output_scitype(::Type{<:ANOVAFilter}) = Table(Continuous)



