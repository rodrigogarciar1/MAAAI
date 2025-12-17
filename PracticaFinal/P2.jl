using CSV
using DataFrames

# ------------------------------------------------
# ------------- Tratamiento de datos -------------
# ------------------------------------------------

function unifyDataset(folderName, fileName)
    fileData = DataFrame()
    for (path, _, files) in walkdir(folderName)
        for file in files
            fileData = vcat(fileData, CSV.read(joinpath(path, file), DataFrame), cols= :union)
        end
    end
    sort!(fileData)
    CSV.write(fileName, fileData)
end

function separateDataframe(df)
    return df[:,1], df[:,2:end-1], df[:,end]
end

function replaceWithMean!(dataMatrix, subjects)
    for subject in unique(subjects)
        subjectData = dataMatrix[findall(x -> x==subject, subjects),:]
        for c in eachcol(subjectData)
            nonMissingRows = findall(x-> !ismissing(x), c)
            columnMean = sum(c[nonMissingRows])/length(dataMatrix[:,1])
            replace!(c, missing => columnMean)
        end
        dataMatrix[findall(x -> x==subject, subjects),:] = subjectData
    end
    return Array(dataMatrix)
end

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    """Convierte un array de características categóricas en una representación
    one-hot.
    Si hay solo dos clases, devuelve una matriz con una columna. Para más de
    dos clases, devuelve una matriz con una columna por clase."""
    if length(classes) <= 2
        # Si solo hay dos clases, se genera una matriz con una columna
        feature = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se genera una matriz con una columna por clase
        oneHot = convert(BitArray{2}, hcat([instance.==classes for instance in feature]...)');
        feature = oneHot;
    end;
    return feature;
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    """Versión sobrecargada de `oneHotEncoding` que automáticamente detecta 
    las clases únicas."""
    return oneHotEncoding(feature, unique(feature));
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    """Versión sobrecargada de `oneHotEncoding` que devuelve el vector dado 
    en forma de columna."""
    return reshape(feature, :, 1);
end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    """Calcula los valores mínimo y máximo de cada columna del dataset."""
    return (minimum(dataset, dims=1), maximum(dataset, dims=1));
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    """Calcula la media y variación estandar de cada columna del dataset."""
    return (mean(dataset, dims=1), std(dataset, dims=1));
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    """Normaliza el dataset utilizando Min-Max Scaling de manera in-place.
    Si un atributo tiene el mismo valor en toda la columna, se ajusta a 0.
    """
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    
    dataset .-= minValues; # Le resta a todo el dataset el valor mínimo por columnas 
    dataset ./= (maxValues .- minValues) .+ 1e-7; # Divide por columnas entre max-min.
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
    
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    """Sobercarga de la función anterior que calcula maxValues y MinValues
    automáticamente."""
    return normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset_copy = copy(dataset);
    normalizeMinMax!(dataset_copy, normalizationParameters);
    return dataset_copy;
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    dataset_copy = copy(dataset);
    normalizeMinMax!(dataset_copy);
    return dataset_copy;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function holdOut(N::Int, P::Real) 
    Lista = randperm(N)
    applied_percentages = floor(Int, round(N * P))
    test = Lista[1:applied_percentages]
    training = Lista[applied_percentages+1:N]
    return (training, test)
end;

function crossvalidation(N::Int64, k::Int64)
    array = collect(1:k);
    array2 = repeat(array, ceil(Int, N/k));
    array2 = array2[1:N];
    shuffle!(array2);
    return array2;
end;

function crossvalidation(feature::AbstractArray{Bool,1}, k::Int64)
    array = collect(1:length(feature));
    array[findall(x->x==true, feature)] = crossvalidation(count(feature.==true), k);
    array[findall(x->x==false, feature)] = crossvalidation(count(feature.==false), k);
    return array;
end;

function crossvalidation(feature::AbstractArray{Bool,2}, k::Int64)
    array = collect(1:size(feature,1));
    for j in eachcol(feature)
        array[findall(x->x==true, j)] = crossvalidation(sum(j), k);
    end;
    return array;
end;

function crossvalidation(feature::AbstractArray{<:Any,1}, k::Int64)
    crossvalidation(oneHotEncoding(feature), k);
end;

function individualWiseFoldCrossValidation(individuals, data, targets, folds)
    crossValidationSubjects = crossvalidation(length(unique(individuals)),folds)

    println(crossValidationSubjects)

    foldTrainData = []
    foldTrainTargets = []
    foldValData = []
    foldValTargets = []

    for i in 1:folds
        foldValUniqueIndividualIndices = findall(x-> x == i, crossValidationSubjects)
        foldValIndividuals = unique(individuals)[foldValUniqueIndividualIndices]
        println("Individuos", foldValIndividuals)

        valIndices = findall(x-> x in foldValIndividuals, individuals)
        trainIndices = findall(x-> !(x  in foldValIndividuals), individuals)
        println(valIndices)
        push!(foldValData, data[valIndices, :])
        push!(foldValTargets, targets[valIndices, :])

        push!(foldTrainData, data[trainIndices, :])
        push!(foldTrainTargets, targets[trainIndices, :])
    end

    return foldTrainData, foldTrainTargets, foldValData, foldValTargets
end;

# -----------------------------------------
# ----------- MinMaxNormalizer ------------
# -----------------------------------------

using MLJModelInterface
using MLJBase
using Tables
using Statistics
import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype, predict
using CategoricalArrays


mutable struct MinMaxNormalizer <: Unsupervised
    mins::Vector{Float64}
    maxs::Vector{Float64}
end

MinMaxNormalizer() = MinMaxNormalizer(Float64[], Float64[])


function fit(model::MinMaxNormalizer, verbosity::Int, X)
    Xmat = Float64.(MLJBase.matrix(X))

    mins = mapslices(minimum, Xmat; dims=1) |> vec
    maxs = mapslices(maximum, Xmat; dims=1) |> vec

    fitresult = (mins, maxs)
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function transform(
    model::MinMaxNormalizer,
    fitresult,
    X
)
    mins, maxs = fitresult
    Xmat = MLJBase.matrix(X)

    Xscaled = (Xmat .- mins') ./ (maxs' .- mins')
    return MLJBase.table(Xscaled)
end

MLJModelInterface.input_scitype(::Type{MinMaxNormalizer}) = Table(Continuous)
MLJModelInterface.output_scitype(::Type{MinMaxNormalizer}) = Table(Continuous)


# --- Definición de transformador ANOVA SelectKBest ---
mutable struct ANOVAFilter <: MLJModelInterface.Supervised
    k::Int
end

# Constructor con argumentos por nombre (estilo MLJ)
ANOVAFilter(; k::Int=2) = ANOVAFilter(k)

import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype

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
            # # Crear test ANOVA
            # test = OneWayANOVATest(groups...)
            # # Nota: según la versión de HypothesisTests, los campos internos pueden variar.
            # # Aquí seguimos el patrón del ejemplo proporcionado por el usuario.
            # MSt = test.SStᵢ / test.DFt  # Mean square treatment (between)
            # MSe = test.SSeᵢ / test.DFe  # Mean square error (within)
            # fstats[j] = MSt / MSe
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

function transform(model::ANOVAFilter, fitresult, X)
    # Convertir X a matriz
    Xmat = MLJBase.matrix(X)
    # Seleccionar columnas usando fitresult (no cache)
    X_selected = Xmat[:, fitresult.idxs]
    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end

input_scitype(::Type{<:ANOVAFilter}) = Table(Continuous)
target_scitype(::Type{<:ANOVAFilter}) = AbstractVector{<:Finite}
output_scitype(::Type{<:ANOVAFilter}) = Table(Continuous)




using InformationMeasures

# ===============================
# ANOVA SelectKBest Filter
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
    
    # Seleccionar columnas usando fitresult (no cache)
    X_selected = Xmat[:, fitresult.idxs]
    
    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end     

# Definir los tipos de entrada y salida
input_scitype(::Type{<:MutualInformation}) = Table(Continuous)
target_scitype(::Type{<:MutualInformation}) = AbstractVector{<:Finite}
output_scitype(::Type{<:MutualInformation}) = Table(Continuous)


# ===============================
# ANOVA SelectKBest Filter
# ===============================
# Definir la estructura del modelo
mutable struct RFE <: MLJModelInterface.Supervised
    k::Int
end

# Constructor con argumentos por nombre (estilo MLJ)
RFE(; k::Int=2) = RFE(k)

# Se hace la importación de las funciones necesarias para poder sobrecargarlas
import MLJModelInterface: fit, transform, input_scitype, target_scitype, output_scitype
using MLJLinearModels
# Definir la función fit
function fit(model::RFE, verbosity::Int, X, y)
    # Convertir X a matriz y asegurarse que es Float64
    Xmat = Float64.(MLJBase.matrix(X))
    
    # Convertir y a vector numérico si es categórico
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)
    modelo = LogistigRegression(0.5)
    # Calcular F-statistics usando HypothesisTests

    n_features = size(Xmat, 2)
    fstats = zeros(Float64, n_features)

    while true
        theta = MLJLinearModels.fit(modelo, Xmat, y_numeric)
        cut = min(Int(length(theta)/2) +1, length(theta)-k +1)
        worstColumns = sortperm(theta, rev=true)[cut:end]
        fstats[worstColumns] = -1

        if length(theta)>k break; end

        Xmat = Xmat[:, 1:end .≠ eachcol(worstColumns)]
        
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
function transform(model::RFE, fitresult, X)
    # Convertir X a matriz
    Xmat = MLJBase.matrix(X)
    
    # Seleccionar columnas usando fitresult (no cache)
    X_selected = Xmat[:, fitresult.idxs]
    
    # Convertir de vuelta a tabla con nombres apropiados
    return MLJBase.table(X_selected, names=fitresult.selected_names)
end     

# Definir los tipos de entrada y salida
input_scitype(::Type{<:RFE}) = Table(Continuous)
target_scitype(::Type{<:RFE}) = AbstractVector{<:Finite}
output_scitype(::Type{<:RFE}) = Table(Continuous)




# ===================================================
# DEFINICION del VOTINGCLASSIFIER compatible con MLJ
# ===================================================

"""
    VotingClassifier <: Probabilistic

Un clasificador *ensemble* que combina las predicciones de múltiples modelos base utilizando diferentes estrategias de votación.

# Campos
- `models::Vector{Probabilistic}`: Vector de modelos probabilísticos base que se combinarán.  
- `voting::Symbol`: Estrategia de votación, que puede ser `:hard` (votación mayoritaria) o `:soft` (promedio de probabilidades).  
- `weights::Union{Nothing, Vector{Float64}}`: Pesos opcionales para cada modelo. Si se establece como `nothing`, todos los modelos tendrán el mismo peso. Los pesos se normalizan automáticamente para que su suma sea 1.0.

# Ejemplos
```julia
# Pesos iguales (por defecto)
voting_clf = VotingClassifier(
    models=[LogisticClassifier(), DecisionTreeClassifier()],
    voting=:soft
)

# Pesos personalizados (se normalizan automáticamente)
voting_clf = VotingClassifier(
    models=[LogisticClassifier(), DecisionTreeClassifier(), RandomForestClassifier()],
    voting=:hard,
    weights=[5, 3, 2]  # Se normalizarán a [0.5, 0.3, 0.2]
)
```
"""
mutable struct VotingClassifier <: Probabilistic   # Models must be probabilistic, inherited from MLJBase
    models::Vector{Probabilistic}
    voting::Symbol  # :hard or :soft
    weights::Union{Nothing, Vector{Float64}}
end


"""
    VotingClassifier(; models=Probabilistic[], voting=:hard, weights=nothing)
Constructor del `VotingClassifier`.

# Argumentos
- `models::Vector{Probabilistic}=Probabilistic[]`: Modelos base que se combinarán.  
- `voting::Symbol=:hard`: Estrategia de votación (`:hard` o `:soft`).  
- `weights::Union{Nothing, Vector{<:Real}}=nothing`: Pesos asignados a cada modelo. Se normalizan automáticamente para que su suma sea 1.0.

# Excepciones
- `AssertionError`: Si el parámetro `voting` no es `:hard` ni `:soft`.  
- `AssertionError`: Si la longitud del vector de pesos no coincide con el número de modelos.  
- `AssertionError`: Si todos los pesos son cero o negativos.
"""

function VotingClassifier(; models=Probabilistic[], voting=:hard, weights=nothing)
    @assert voting in [:hard, :soft] "The only possible labels are :hard or :soft"
    
    normalized_weights = nothing
    if weights !== nothing
        @assert length(weights) == length(models) "El número de pesos tiene que ser igual al de modelos"
        @assert all(w >= 0 for w in weights) "Todos los pesos deben de ser no negativos"
        
        # Suma normalizada de los pesos 1.0
        normalized_weights = Float64.(weights) ./ sum(weights)
    end
    
    return VotingClassifier(models, voting, normalized_weights)
end

"""
    MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)

Entrena el `VotingClassifier` ajustando cada modelo base con los datos proporcionados.

# Argumentos
- `model::VotingClassifier`: Instancia del clasificador de votación.  
- `verbosity::Int`: Nivel de verbosidad para el registro del proceso de entrenamiento.  
- `X`: Características de entrenamiento (en formato de tabla).  
- `y`: Variable objetivo de entrenamiento (vector categórico).

# Retorna
- `fitresults`: Vector de máquinas entrenadas (una por cada modelo base).  
- `cache`: `nothing` (no se implementa almacenamiento en caché).  
- `report`: Tupla con nombre que contiene información del entrenamiento (número de modelos, estrategia de votación y pesos normalizados).
"""

function MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)
    # Entrenar cada modelos base
    machs = [begin
        mm = machine(deepcopy(m), X, y)
        fit!(mm, verbosity=0)
        mm
    end for m in model.models]

    fitresults = (
        machines = machs,
        class_levels = collect(levels(y)),   #Mantener las clases para futuras ejecuciones
        class_pool = CategoricalArrays.pool(y)
    )
    
    # Guardar la información necesaria
    cache = nothing
    report = (n_models=length(model.models), voting=model.voting, weights=model.weights)
    
    return fitresults, cache, report
end


"""
    MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)

Predice las etiquetas de clase utilizando **votación dura** (*hard voting*), es decir, votación mayoritaria con pesos opcionales.

# Argumentos
- `model::VotingClassifier`: Instancia del clasificador de votación.  
- `fitresult`: Vector de máquinas entrenadas obtenido en la fase de ajuste.  
- `Xnew`: Nuevos datos sobre los que se realizará la predicción.

# Retorna
- Vector categórico con las etiquetas de clase predichas, calculadas mediante votación mayoritaria (ponderada o no).

# Detalles
Cada modelo base emite un voto por una clase.  
Si se han definido pesos, cada voto se multiplica por el peso correspondiente.  
La clase con el mayor número de votos (ponderados) es seleccionada como predicción final.
"""

function MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)
    machines = fitresult.machines
    class_levels = fitresult.class_levels
    
    # Obtener las prediciones de todos los modelos
    predictions = [categorical(predict_mode(mach, Xnew), levels=class_levels) for mach in machines]
    
    # Recoger datos básicos de la simulación
    n_samples = length(predictions[1])
    n_models = length(machines)
    
    # Establecer todos los pesos iguales, si no se han especificado
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights
    
    # Output Vector with the same type as pthe predictions
    ensemble_pred = similar(predictions[1])
    
    for i in 1:n_samples
        # Contar el número de votos por clase
        vote_counts = Dict{eltype(predictions[1][1]), Float64}()
        
        for (j, prediction) in enumerate(predictions)
            vote_counts[prediction[i]] = get(vote_counts, prediction[i], 0.0) + weights[j]
        end
        
        # Cambio necesario para problemas binarios (sin usar argmax sobre Dict)
        best_label = nothing
        best_score = -Inf
        for (lbl, sc) in vote_counts
            if sc > best_score
                best_score = sc
                best_label = lbl
            end
        end

        ensemble_pred[i] = best_label
    end

    return ensemble_pred
end

"""
    MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)

Predice las probabilidades de clase utilizando la estrategia de votación especificada.

# Argumentos
- `model::VotingClassifier`: Instancia del clasificador de votación.  
- `fitresult`: Vector de máquinas entrenadas obtenido durante el ajuste.  
- `Xnew`: Nuevos datos sobre los que se realizarán las predicciones.

# Retorna
- Vector de distribuciones `UnivariateFinite` que representan las probabilidades de pertenencia a cada clase.

# Detalles
- Para la votación `:hard`: se devuelven predicciones deterministas encapsuladas en `UnivariateFinite` (con pesos opcionales).  
- Para la votación `:soft`: se calculan las probabilidades promediando las distribuciones generadas por todos los modelos base, aplicando los pesos correspondientes.
"""

function MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)
    machines     = fitresult.machines
    class_levels = fitresult.class_levels
    class_pool   = fitresult.class_pool

    result = if model.voting == :hard
       # Hard Voting
        yhat = MLJModelInterface.predict_mode(model, fitresult, Xnew)
        yhat = categorical(yhat; levels=class_levels)  # asegura mismos niveles

        # Devuelve las probabilidades como one-hot encoded 
        [MLJBase.UnivariateFinite(
                    class_levels,
                    [lvl == yhat[i] ? 1.0 : 0.0 for lvl in class_levels];
                    pool=class_pool
                ) for i in 1:length(yhat)]
    else
        # Soft voting
        all_predictions = [predict(mach, Xnew) for mach in machines]

        n_samples = length(all_predictions[1])
        n_models  = length(machines)
        n_classes = length(class_levels)
        weights   = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

        avg_probs = zeros(n_samples, n_classes)
        for (w, prediction) in zip(weights, all_predictions)
            for i in 1:n_samples
                p_i = prediction[i]
                if p_i isa MLJBase.UnivariateFinite
                    for (j, level) in enumerate(class_levels)
                        avg_probs[i, j] += w * pdf(p_i, level)
                    end
                else
                    # determinista -> one-hot
                    for (j, level) in enumerate(class_levels)
                        avg_probs[i, j] += w * (p_i == level ? 1.0 : 0.0)
                    end
                end
            end
        end

        # Normalizar cada probabilidad para evitar problemas con el redondeo con los números reales
        for i in 1:n_samples
            s = sum(@view avg_probs[i, :])
            if s > 0
                @. avg_probs[i, :] = avg_probs[i, :] / s
            end
        end

        # Usa la misma codificación entre llamadas para prevenir confusiones
        [MLJBase.UnivariateFinite(class_levels, @view avg_probs[i, :]; pool=class_pool)
         for i in 1:n_samples]
    end

    return result
end
"""
Registro de metadatos del modelo para `VotingClassifier`.

Especifica los tipos de entrada/salida y las capacidades para su integración con MLJ.
"""
MLJModelInterface.metadata_model(VotingClassifier,
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    load_path="VotingClassifier"
)