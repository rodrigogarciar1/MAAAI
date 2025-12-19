using MLJModelInterface
using MLJBase
using Tables
using CategoricalArrays
using MLJLinearModels

# Definir la estructura del modelo
mutable struct RFE <: MLJModelInterface.Supervised
    k::Int               # número de variables a conservar
end

# Constructor con argumentos por nombre (estilo MLJ)
RFE(; k::Int=2) = RFE(k)

# Se hace la importación de las funciones necesarias para poder sobrecargarlas
import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype
import MLJModelInterface: Table, Continuous, Finite

# -----------------------
# FIT: entrenamiento RFE
# -----------------------
function fit(model::RFE, verbosity::Int, X, y)

    # Convertimos X a matriz de Float64
    Xmat = Float64.(MLJBase.matrix(X))
    n_features = size(Xmat, 2)

    # Convertir y a vector numérico si es categórico
    y_vec =
        if y isa CategoricalVector
            Int.(levelcode.(y))
        else
            collect(y)
        end

    # Modelo de regresión logística de MLJLinearModels
    logreg = LogisticRegression(lambda=0.5)

    # Índices de características activos
    active = collect(1:n_features)

    # Bucle RFE: eliminar el 50 % de las variables en cada iteración
    while length(active) > model.k
        # Entrenamos el modelo solo con las columnas activas
        θ = MLJLinearModels.fit(logreg, Xmat[:, active], y_vec)

        # primer coeficiente lo excluimos
        coefs = θ[2:end]

        n_current = length(active)
        # Número de variables 50 % de las actuales

        n_to_remove = max(1, floor(Int, n_current/2))
        n_to_remove = min(n_to_remove, n_current - model.k)

        # Ordenamos por |coef| de menor a mayor 
        order_local = sortperm(abs.(coefs); rev=false)
        worst_local = order_local[1:n_to_remove]

        # Pasamos estos índices locales a índices globales de columnas
        worst_global = active[worst_local]

        # Actualizamos el conjunto de columnas activas
        active = setdiff(active, worst_global)
    end

    # Índices finales seleccionados
    idxs = sort(active)

    # Guardar los nombres de las columnas originales
    feature_names = collect(Tables.columnnames(X))
    selected_names = feature_names[idxs]

    # fitresult debe contener la info necesaria para transform
    fitresult = (idxs = idxs, selected_names = selected_names)
    cache = nothing
    report = (idxs = idxs, selected_features = selected_names)

    return fitresult, cache, report
end

# -----------------------
# TRANSFORM: aplicar RFE
# -----------------------
function transform(model::RFE, fitresult, X)
    # Convertir X a matriz
    Xmat = MLJBase.matrix(X)

    # Seleccionar columnas usando los índices aprendidos
    X_selected = Xmat[:, fitresult.idxs]

    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end     

# -----------------------
# Tipos de entrada y salida (scitypes)
# -----------------------
input_scitype(::Type{<:RFE}) = Table(Continuous)
target_scitype(::Type{<:RFE}) = AbstractVector{<:Finite}
output_scitype(::Type{<:RFE}) = Table(Continuous)
