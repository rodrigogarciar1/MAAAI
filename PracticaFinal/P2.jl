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
    dataset ./= (maxValues .- minValues); # Divide por columnas entre max-min.
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