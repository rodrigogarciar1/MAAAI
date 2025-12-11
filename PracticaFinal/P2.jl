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

function replaceWithMean!(dataMatrix)
    for c in eachcol(dataMatrix)
        nonMissingRows = findall(x-> !ismissing(x), c)
        columnMean = sum(c[nonMissingRows])/length(dataMatrix[:,1])
        replace!(c, missing => columnMean)
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