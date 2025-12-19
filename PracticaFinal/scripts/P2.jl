using Pkg
Pkg.activate("../environment")
Pkg.instantiate()

using CSV
using DataFrames
using MLJBase
using MLJModels
using MLJModelInterface

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

using Random

import CategoricalArrays: pool

mutable struct VotingClassifier <: Probabilistic
    models::Vector{<:Probabilistic}
    voting::Symbol
    weights::Union{Nothing, Vector{Float64}}
end

function VotingClassifier(; models=Probabilistic[], voting=:hard, weights=nothing)
    @assert voting in [:hard, :soft] "Voting debe ser :hard o :soft"
    
    if weights !== nothing
        @assert length(weights) == length(models) "Pesos y modelos deben tener la misma longitud"
        @assert all(w >= 0 for w in weights) "Los pesos deben ser no negativos"
        weights = weights ./ sum(weights) # Normalización
    end
    
    return VotingClassifier(models, voting, weights)
end


function MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)
    n_models = length(model.models)
    n_rows = nrows(X)
    

    Random.seed!(104)

    indexes = crossvalidation(n_rows, n_models)
    
    machs = []
    for m in 1:n_models
        idx = findall(x -> x == m, indexes)
        X_sub = selectrows(X, idx)
        y_sub = y[idx]
        
        mach = machine(model.models[m], X_sub, y_sub)
        fit!(mach, verbosity=verbosity)
        push!(machs, mach)
    end

    fitresults = (
        machines = machs,
        class_levels = levels(y),
        class_pool = pool(y)
    )
    
    cache = nothing
    report = (n_models=n_models, voting=model.voting, weights=model.weights)
    
    return fitresults, cache, report
end
-

function MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)
    machines = fitresult.machines
    class_levels = fitresult.class_levels
    n_samples = nrows(Xnew)
    n_models = length(machines)
    
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights
    
    all_preds = [predict_mode(mach, Xnew) for mach in machines]
    
    ensemble_pred = CategoricalArray{eltype(class_levels)}(undef, n_samples)
    levels!(ensemble_pred, class_levels)

    for i in 1:n_samples
        vote_counts = Dict{Any, Float64}()
        for m in 1:n_models
            label = all_preds[m][i]
            vote_counts[label] = get(vote_counts, label, 0.0) + weights[m]
        end
        

        best_label = first(keys(vote_counts))
        max_v = -1.0
        for (lbl, v) in vote_counts
            if v > max_v
                max_v = v
                best_label = lbl
            end
        end
        ensemble_pred[i] = best_label
    end

    return ensemble_pred
end


function MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)
    if model.voting == :hard
        yhat = predict_mode(model, fitresult, Xnew)
        return MLJBase.UnivariateFinite(fitresult.class_levels, yhat)
    end


    machines = fitresult.machines
    n_models = length(machines)
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

    all_probs = [predict(mach, Xnew) for mach in machines]

    combined_probs = weights[1] * all_probs[1]
    for m in 2:n_models
        combined_probs += weights[m] * all_probs[m]
    end
    
    return combined_probs
end


MLJModelInterface.metadata_model(VotingClassifier,
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    load_path="VotingClassifier"
)