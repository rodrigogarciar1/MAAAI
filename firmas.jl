
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images
using Flux.Losses

function fileNamesFolder(folderName::String, extension::String)
    extension = uppercase(extension); 
    fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName))
    
    return vec(String.(chop.(fileNames, tail = length(extension)+1))) #Elimina el ".*" del nombre de todos los archivos encontrados
end;




function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float32)

    datasetName *=".tsv"; #Añade la extension
    path = abspath(joinpath(datasetFolder, datasetName)) #Obtiene la ruta del archivo
    datasetMatrix = Matrix #Creación de objeto vacio

    try
    datasetMatrix = readdlm(path, '\t')
    catch
        println("ERROR: FILE NOT FOUND")
        return nothing
    end;

    targetColumn = findfirst(datasetMatrix[1,:].=="target"); #Encuentra la columna de targets
    datasetMatrix = datasetMatrix[2:end,:]; #Elimina la fila de etiquetas de la matriz
    targets = vec(reshape(Bool.(datasetMatrix[:,targetColumn]), 1, :)); #Convierte la columna de targets a booleanos y lo convierte en array
    inputs = datasetMatrix[:, (1:end) .!= targetColumn]; #Todas las colmunas menos la de target
    
    return (convert.(datasetType ,inputs), targets)
end;



function loadImage(imageName::String, datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)

    imageName *= ".tif"; #extension
    path = abspath(joinpath(datasetFolder, imageName)); #ruta del archivo
    ImageMatrix = Matrix; #matriz de la imagen vacia

    if !isfile(path)
        println("ERROR: FILE NOT FOUND");
        return nothing;
    end;
    image = load(path); 
    image = imresize(image, (resolution,resolution)) #cambiar la resolución y a grises 1 y 0

    #comprobar si ya está en grises
    if eltype(image) <: Gray
        ImageMatrix = Float64.(image); #si esta en grises se pasa a números
    else

        ImageMatrix = gray.(image) #convertir a escala de grises con 1 o 0
    end;

    ImageMatrix = convert(Matrix{datasetType}, ImageMatrix); #convertir en el tipo necesario

    return ImageMatrix
end;


function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)

    #Obtener el nombre de los archivos y añade la extension
    imagesNames = fileNamesFolder(datasetFolder, "tif")
    #Cargar todas las imagenes mediante un broadcast
    images = loadImage.(imagesNames, datasetFolder; datasetType=datasetType, resolution=resolution)
    #Convertir las imagenes a NCHW
    imagesNCHW = convertImagesNCHW(images)

    return imagesNCHW
end;


showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));



function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)

    # Cargar el archivo MNIST.jld2
    mnistPath = abspath(joinpath(datasetFolder, "MNIST.jld2"))
    mnistData = load(mnistPath)
    
    # Extraer datos del diccionario
    trainImages = mnistData["train_imgs"]
    trainLabels = mnistData["train_labels"]
    testImages = mnistData["test_imgs"] 
    testLabels = mnistData["test_labels"]
    
    # Si labels contiene -1, cambiar las etiquetas no incluidas a -1
    if -1 in labels
        trainLabels[.!in.(trainLabels, [setdiff(labels, -1)])] .= -1
        testLabels[.!in.(testLabels, [setdiff(labels, -1)])] .= -1
    end
    
    # Obtener índices de las imágenes que queremos mantener
    trainIndices = in.(trainLabels, [labels])
    testIndices = in.(testLabels, [labels])
    
    # Filtrar imágenes y etiquetas
    filteredTrainImages = trainImages[trainIndices]
    filteredTrainLabels = trainLabels[trainIndices]
    filteredTestImages = testImages[testIndices]
    filteredTestLabels = testLabels[testIndices]
    
    # Convertir imágenes a formato NCHW
    trainImagesNCHW = convertImagesNCHW(filteredTrainImages)
    testImagesNCHW = convertImagesNCHW(filteredTestImages)
    
    # Convertir al tipo de dato especificado
    trainImagesNCHW = convert.(datasetType, trainImagesNCHW)
    testImagesNCHW = convert.(datasetType, testImagesNCHW)
    
    return (trainImagesNCHW, filteredTrainLabels, testImagesNCHW, filteredTestLabels)
end;


function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end;


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    m = intervalDiscreteVector(data)
    unique_vals = sort(unique(data)) #los valores no son únicos, hay mas angulos, mejor que la media usar esto para ordenar todo
    n = length(unique_vals)  #nmero de valores distintos

    if m == 0.0
        #caso continuo: normalización en [0, 2π]
        angulos = (data .- minimum(data)) ./ (maximum(data) - minimum(data)) .* 2π
    else
        #posición del valor en la lista ordenada * 2π/n
        paso = 2π / n
        #posición de cada valor en la lista
        angulos = [ (findfirst(==(x), unique_vals) - 1) * paso for x in data ]
    end

    senos = sin.(angulos)
    cosenos = cos.(angulos)

    return senos, cosenos
end;




function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    targets = readdlm(abspath(joinpath(datasetFolder, "elec2_label.dat")))
    inputs = readdlm(abspath(joinpath(datasetFolder, "elec2_data.dat")))

    targets = vec(Bool.(targets))

    inputs = inputs[:, (1:end) .∉ ((1,4),)]
    (senos, cosenos) = cyclicalEncoding(inputs[:,1])
    inputs = inputs[:, (2:end)]

    return (hcat(senos, cosenos, inputs), targets)

end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    # Crea una RNA sin capas ocultas según el número de salidas
    if numOutputs > 2
        return Chain(Dense(numInputs, numOutputs, identity), softmax);
    else
        return Chain(Dense(numInputs, 1, σ));
    end;
end;

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    # Referenciar la capa de salida de RNA y las capas previas
    outputLayer = previousANN[indexOutputLayer(previousANN)];
    previousLayers = previousANN[1:(indexOutputLayer(previousANN)-1)];
    
    # Obtener el número de entradas y salidas a partir de la capa de salida
    numInputsOutputLayer = size(outputLayer.weight, 2);
    numOutputsOutputLayer = size(outputLayer.weight, 1);
    
    # Crear  una  RNA  nueva en base al numero de salidas
    if numOutputsOutputLayer > 2
        ann = Chain(
            previousLayers...,
            SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx)),
            Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity),
            softmax
        );
    else
        ann = Chain(
            previousLayers...,
            SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx)),
            Dense(numInputsOutputLayer + 1, 1, σ)
        );
    end;
    
    # Obtener la nueva capa de salida
    newOutputLayer = ann[indexOutputLayer(ann)];
    
    # Copiar pesos y bias de la red anterior
    # Poner la última columna (conexiones de la nueva neurona) a 0
    newOutputLayer.weight[:, end] .= 0;
    # Copiar los pesos anteriores
    newOutputLayer.weight[:, 1:end-1] .= outputLayer.weight;
    # Copiar el bias
    newOutputLayer.bias .= outputLayer.bias;
    
    return ann;
end;

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    inputs, targets = trainingDataset; # dataset = Tuple(inputs, targets)
    inputs = convert(AbstractArray{Float32,2}, inputs); # Se pasan todos los valores a Float32 para mayor velocidad

    numEpoch = 0

    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    trainingLosses = [loss(ann, inputs, targets)]; # Lista con los valores de error
    opt_state = Flux.setup(Adam(learningRate), ann);
    lossChange = 1

    while (numEpoch < maxEpochs) & (trainingLosses[end] > minLoss) & (lossChange > minLossChange)
        numEpoch+=1

        if trainOnly2LastLayers
            Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)]);
        end

        Flux.train!(loss, ann, [(inputs, targets)], opt_state);
        Loss = loss(ann, inputs, targets)
        append!(trainingLosses, Loss);

        if numEpoch >= lossChangeWindowSize
            lossWindow = trainingLosses[end-lossChangeWindowSize+1:end]; 
            minLossValue, maxLossValue = extrema(lossWindow); 
            lossChange = ((maxLossValue-minLossValue)/minLossValue) 
        end

    end

    return convert.(Float32, trainingLosses)
end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    inputs, outputs = trainingDataset;
    inputs = transpose(inputs); #trasponer las funciones
    outputs = transpose(outputs);

    inputs = convert(Array{Float32}, inputs) #matriz de entradas a float32

    ann = newClassCascadeNetwork(size(inputs,1), size(outputs,1));

    loss = trainClassANN!(ann,(inputs, outputs),false ,
                maxEpochs = maxEpochs, minLoss= minLoss, 
                learningRate= learningRate, minLossChange=minLossChange, 
                lossChangeWindowSize = lossChangeWindowSize);
    
    loss_total = convert(Array{Float32,1}, loss)
    
    for neuron in 1:maxNumNeurons
        ann = addClassCascadeNeuron(ann, transferFunction=transferFunction)
        if neuron >= 1
            if neuron > 1
                loss_partial =  trainClassANN!(ann,(inputs, outputs),true ,
                maxEpochs = maxEpochs, minLoss= minLoss, 
                learningRate= learningRate, minLossChange=minLossChange, 
                lossChangeWindowSize = lossChangeWindowSize);

                loss_total = vcat(loss_total, convert(Array{Float32,1}, loss_partial[2:end]))


            end;

            loss_global = trainClassANN!(ann,(inputs, outputs),false ,
                                        maxEpochs = maxEpochs, minLoss= minLoss, 
                                        learningRate= learningRate, minLossChange=minLossChange, 
                                        lossChangeWindowSize = lossChangeWindowSize);
            loss_total = vcat(loss_total, convert(Array{Float32,1}, loss_global[2:end]))

        end;
    end;


    return tuple(ann,loss_total)
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    inputs, outputs = trainingDataset;
    outputs_matriz = reshape(outputs, (length(outputs),1))
    return trainClassCascadeANN(maxNumNeurons, (inputs, outputsMatrix),
                               transferFunction=transferFunction,maxEpochs=maxEpochs,minLoss=minLoss,
                               learningRate=learningRate, minLossChange=minLossChange,
                               lossChangeWindowSize=lossChangeWindowSize)
    
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    #
    # Codigo a desarrollar
    #
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    #
    # Codigo a desarrollar
    #
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;
function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;
function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;





function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    #
    # Codigo a desarrollar
    #
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    #
    # Codigo a desarrollar
    #
end;

function randomImages(numImages::Int, resolution::Int)
    #
    # Codigo a desarrollar
    #
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = MLJ.predict(model, MLJ.table(inputs));



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

