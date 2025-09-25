

# Archivo de pruebas para realizar autoevaluación de algunas funciones de los ejercicios

# Importamos el archivo con las soluciones a los ejercicios
include("firmas.jl");
#   Cambiar "soluciones.jl" por el nombre del archivo que contenga las funciones desarrolladas



# Fichero de pruebas realizado con la versión 1.11.2 de Julia
println(VERSION)
#  y la 1.11.3 de Random
using Random; println(Random.VERSION)
#  y la versión 0.14.25 de Flux
import Pkg
Pkg.status("Flux")

# Es posible que con otras versiones los resultados sean distintos, estando las funciones bien, sobre todo en la funciones que implican alguna componente aleatoria

# Para la correcta ejecución de este archivo, los datasets estarán en las siguientes carpetas:
datasetFolder = "./datasets"; # Incluye el dataset MNIST
imageFolder = "./datasets/images";
# Cambiadlas por las carpetas donde tengáis los datasets y las imágenes

@assert(isdir(datasetFolder))
@assert(isdir(imageFolder))

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------


# imageFileNames = fileNamesFolder(imageFolder,"tif");
# @assert(imageFileNames == ["cameraman", "lake", "lena_gray_512", "livingroom", "mandril_gray", "peppers_gray", "pirate", "walkbridge"]);

# println("==============================")
# println("Tests fileNamesFolder pasados")
# println("==============================")
# println()

# inputs, targets = loadDataset("sonar", datasetFolder; datasetType=Float32);
# @assert(size(inputs)==(208,60))
# @assert(length(targets)==208)
# @assert(eltype(inputs)==Float32)
# @assert(eltype(targets)==Bool)

# println("==============================")
# println("Tests loadDataset pasados")
# println("==============================")
# println()

# image = loadImage("cameraman", imageFolder; datasetType=Float64, resolution=64)
# @assert(size(image)==(64,64))
# @assert(eltype(image)==Float64)

# println("==============================")
# println("Tests loadImage pasados")
# println("==============================")
# println()

# imagesNCHW = loadImagesNCHW(imageFolder; datasetType=Float64, resolution=32)
# @assert(size(imagesNCHW)==(8,1,32,32))
# @assert(eltype(imagesNCHW)==Float64)

# println("==============================")
# println("Tests loadImagesNCHW pasados")
# println("==============================")
# println()

# MNISTDataset = loadMNISTDataset(datasetFolder; labels=[3,6,9], datasetType=Float64)
# @assert(size(MNISTDataset[1])==(17998, 1, 28, 28))
# @assert(eltype(MNISTDataset[1])==Float64)
# @assert(length(MNISTDataset[2])==17998)
# @assert(sort(unique(MNISTDataset[2]))==[3,6,9])
# @assert(size(MNISTDataset[3])==(2977, 1, 28, 28))
# @assert(eltype(MNISTDataset[3])==Float64)
# @assert(length(MNISTDataset[4])==2977)
# @assert(sort(unique(MNISTDataset[4]))==[3,6,9])


# MNISTDataset = loadMNISTDataset(datasetFolder; labels=[2,7,-1], datasetType=Float32)
# @assert(size(MNISTDataset[1])==(60000, 1, 28, 28))
# @assert(eltype(MNISTDataset[1])==Float32)
# @assert(length(MNISTDataset[2])==60000)
# @assert(eltype(MNISTDataset[2])<:Integer)
# @assert(sort(unique(MNISTDataset[2]))==[-1,2,7])
# @assert(size(MNISTDataset[3])==(10000, 1, 28, 28))
# @assert(eltype(MNISTDataset[3])==Float32)
# @assert(length(MNISTDataset[4])==10000)
# @assert(eltype(MNISTDataset[4])<:Integer)
# @assert(sort(unique(MNISTDataset[4]))==[-1,2,7])

# println("==============================")
# println("Tests loadMNISTDataset pasados")
# println("==============================")
# println()

# sinEncoding, cosEncoding = cyclicalEncoding([1, 2, 3, 2, 1, 0, -1, -2, -3]);
# @assert(all(isapprox.(sinEncoding, [-0.433883739117558, -0.9749279121818236, -0.7818314824680299, -0.9749279121818236, -0.433883739117558, 0.43388373911755823, 0.9749279121818236, 0.7818314824680298, 0.0]; rtol=1e-4)))
# @assert(all(isapprox.(cosEncoding, [-0.9009688679024191, -0.2225209339563146, 0.6234898018587334, -0.2225209339563146, -0.9009688679024191, -0.900968867902419, -0.22252093395631434, 0.6234898018587336, 1.0]; rtol=1e-4)))

# println("==============================")
# println("Tests cyclicalEncoding pasados")
# println("==============================")
# println()

# inputs, targets = loadStreamLearningDataset(datasetFolder; datasetType=Float64)
# @assert(size(inputs)==(45312,7))
# @assert(length(targets)==45312)
# @assert(eltype(inputs)==Float64)
# @assert(eltype(targets)==Bool)

# println("==============================")
# println("Tests loadStreamLearningDataset pasados")
# println("==============================")
# println()

# # ----------------------------------------------------------------------------------------------
# # ------------------------------------- Ejercicio 2 --------------------------------------------
# # ----------------------------------------------------------------------------------------------


# using Random:seed!

# # Establecemos la semilla para que los resultados sean siempre los mismos
# # Comprobamos que la generación de números aleatorios es la esperada:
# seed!(1); @assert(isapprox(rand(), 0.07336635446929285))
# #  Si fallase aquí, seguramente dara error al comprobar los resultados de la ejecución de la siguiente función porque depende de la generación de números aleatorios

# inputs, targets = loadDataset("sonar", datasetFolder; datasetType=Float32);



# seed!(1); ann = newClassCascadeNetwork(size(inputs,2), 3)
# @assert(length(ann)==2)
# @assert(size(ann[1].weight)==(3,60))
# @assert(ann[1].σ==identity)
# @assert(ann[2]==softmax)
# @assert(size(ann(inputs'))==(3,208))

# seed!(1); ann = newClassCascadeNetwork(size(inputs,2), 1)
# @assert(length(ann)==1)
# @assert(size(ann[1].weight)==(1,60))
# @assert(ann[1].σ==σ)
# @assert(size(ann(inputs'))==(1,208))


# println("==============================")
# println("Tests newClassCascadeNetwork pasados")
# println("==============================")
# println()

# seed!(1); newAnn = addClassCascadeNeuron(ann; transferFunction=tanh)
# @assert(length(newAnn)==2)
# @assert(isa( newAnn[1], SkipConnection))
# @assert(isa( newAnn[1].layers, Dense))
# @assert(size(newAnn[1].layers.weight)==(1,60))
# @assert(     newAnn[1].layers.σ == tanh)
# @assert(isa( newAnn[2], Dense))
# @assert(size(newAnn[2].weight)==(1,61))
# @assert(     newAnn[2].σ == σ)
# @assert(size(newAnn(inputs'))==(1,208))
# @assert(all(isapprox.(newAnn[2].weight[:,1:end-1], ann[1].weight)))
# @assert(all(  iszero.(newAnn[2].weight[:,end])))
# @assert(all(  iszero.(newAnn[2].bias)))
# @assert(all(isapprox.(newAnn[1].layers.bias, [0.0])));


# seed!(1); newANN = addClassCascadeNeuron(newAnn; transferFunction=σ)
# @assert(length(newANN)==3)
# @assert(isa( newANN[1], SkipConnection))
# @assert(isa( newANN[1].layers, Dense))
# @assert(size(newANN[1].layers.weight)==(1,60))
# @assert(     newANN[1].layers.σ == tanh)
# @assert(isa( newANN[2], SkipConnection))
# @assert(isa( newANN[2].layers, Dense))
# @assert(size(newANN[2].layers.weight)==(1,61))
# @assert(     newANN[2].layers.σ == σ)
# @assert(isa( newANN[3], Dense))
# @assert(size(newANN[3].weight)==(1,62))
# @assert(     newANN[3].σ == σ)
# @assert(size(newANN(inputs'))==(1,208))

# @assert(all(isapprox.(newANN[1].layers.weight, newAnn[1].layers.weight)))
# @assert(all(isapprox.(newANN[1].layers.bias,   newAnn[1].layers.bias)))

# @assert(all(isapprox.(newANN[3].weight[:,1:end-1], newAnn[2].weight)))
# @assert(all(  iszero.(newANN[3].weight[:,end])))
# @assert(all(isapprox.(newANN[3].bias, newAnn[2].bias)))


# println("==============================")
# println("Tests addClassCascadeNeuron pasados")
# println("==============================")
# println()


# trainingLosses = trainClassANN!(newANN, (inputs', reshape(targets, 1, :)), true;
#     maxEpochs=5, minLoss=0.0, learningRate=0.01, minLossChange=1e-6, lossChangeWindowSize=3)
# @assert(eltype(trainingLosses)==Float32);
# @assert(all(isapprox.(trainingLosses, Float32[0.70495284, 0.69371563, 0.68812746, 0.6846968, 0.6808723, 0.67615265])))
# @assert(all(isapprox.(newANN.layers[1].layers.bias, [0.0])));
# @assert(all(isapprox.(newANN.layers[2].layers.bias, [-0.011772233])));
# @assert(all(isapprox.(newANN.layers[3].bias,        [-0.020025687])));


# trainingLosses = trainClassANN!(newANN, (inputs', reshape(targets, 1, :)), false;
#     maxEpochs=5, minLoss=0.0, learningRate=0.01, minLossChange=1e-6, lossChangeWindowSize=3)
# @assert(eltype(trainingLosses)==Float32);
# @assert(all(isapprox.(trainingLosses, Float32[0.67615265, 0.6685693, 0.66276264, 0.6575051, 0.6521533, 0.64666885])))
# @assert(all(isapprox.(newANN.layers[1].layers.bias, [-0.04038628])));
# @assert(all(isapprox.(newANN.layers[2].layers.bias, [-0.04788568])));
# @assert(all(isapprox.(newANN.layers[3].bias,        [0.018007565])));

# println("==============================")
# println("Tests trainClassANN! pasados")
# println("==============================")
# println()

# seed!(1); ann, trainingLosses = trainClassCascadeANN(4, (inputs, reshape(targets, :, 1));
#     transferFunction=tanh, maxEpochs=10, minLoss=0.0, learningRate=0.001, minLossChange=1e-6, lossChangeWindowSize=3)
# @assert(eltype(trainingLosses)==Float32);
# @assert(all(isapprox.(trainingLosses, Float32[0.70495284, 0.7035844, 0.70227885, 0.7010367, 0.6998583, 0.69874185, 0.69768566, 0.69668776, 0.695746, 0.6948588, 0.6940225, 0.6931689, 0.69235694, 0.6915839, 0.690847, 0.69014233, 0.68946385, 0.6888036, 0.68815523, 0.68751377, 0.6868759, 0.68618554, 0.6854976, 0.6848116, 0.6841279, 0.6834463, 0.682767, 0.6820895, 0.6814145, 0.6807416, 0.680071, 0.67938733, 0.6787043, 0.6780223, 0.6773417, 0.6766619, 0.67598337, 0.6753062, 0.67463005, 0.673955, 0.6732813, 0.67262596, 0.67197335, 0.67132294, 0.67067486, 0.6700293, 0.6693859, 0.6687449, 0.668106, 0.66746974, 0.66683555, 0.6661611, 0.66548693, 0.66481334, 0.66413987, 0.6634669, 0.6627942, 0.66212213, 0.6614505, 0.6607793, 0.6601087, 0.6594853, 0.6588651, 0.658248, 0.6576337, 0.6570225, 0.65641433, 0.655809, 0.6552067, 0.65460736, 0.65401053, 0.6533396, 0.6526679, 0.65199506, 0.6513217, 0.65064716, 0.6499717, 0.6492952, 0.6486175, 0.6479387, 0.64725846])))
# @assert(length(ann)==5)
# @assert(all(isapprox.(ann.layers[1].layers.bias, [0.023382332])));
# @assert(all(isapprox.(ann.layers[2].layers.bias, [-0.03736015])));
# @assert(all(isapprox.(ann.layers[3].layers.bias, [-0.028381256])));
# @assert(all(isapprox.(ann.layers[4].layers.bias, [-0.018296173])));
# @assert(all(isapprox.(ann.layers[5].bias,        [0.042160995])));


# seed!(1); ann, trainingLosses = trainClassCascadeANN(4, (inputs, targets);
#     transferFunction=tanh, maxEpochs=10, minLoss=0.0, learningRate=0.001, minLossChange=1e-6, lossChangeWindowSize=3)
# @assert(eltype(trainingLosses)==Float32);
# @assert(all(isapprox.(trainingLosses, Float32[0.70495284, 0.7035844, 0.70227885, 0.7010367, 0.6998583, 0.69874185, 0.69768566, 0.69668776, 0.695746, 0.6948588, 0.6940225, 0.6931689, 0.69235694, 0.6915839, 0.690847, 0.69014233, 0.68946385, 0.6888036, 0.68815523, 0.68751377, 0.6868759, 0.68618554, 0.6854976, 0.6848116, 0.6841279, 0.6834463, 0.682767, 0.6820895, 0.6814145, 0.6807416, 0.680071, 0.67938733, 0.6787043, 0.6780223, 0.6773417, 0.6766619, 0.67598337, 0.6753062, 0.67463005, 0.673955, 0.6732813, 0.67262596, 0.67197335, 0.67132294, 0.67067486, 0.6700293, 0.6693859, 0.6687449, 0.668106, 0.66746974, 0.66683555, 0.6661611, 0.66548693, 0.66481334, 0.66413987, 0.6634669, 0.6627942, 0.66212213, 0.6614505, 0.6607793, 0.6601087, 0.6594853, 0.6588651, 0.658248, 0.6576337, 0.6570225, 0.65641433, 0.655809, 0.6552067, 0.65460736, 0.65401053, 0.6533396, 0.6526679, 0.65199506, 0.6513217, 0.65064716, 0.6499717, 0.6492952, 0.6486175, 0.6479387, 0.64725846])))
# @assert(length(ann)==5)
# @assert(all(isapprox.(ann.layers[1].layers.bias, [0.023382332])));
# @assert(all(isapprox.(ann.layers[2].layers.bias, [-0.03736015])));
# @assert(all(isapprox.(ann.layers[3].layers.bias, [-0.028381256])));
# @assert(all(isapprox.(ann.layers[4].layers.bias, [-0.018296173])));
# @assert(all(isapprox.(ann.layers[5].bias,        [0.042160995])));


# println("==============================")
# println("Tests trainClassCasacadeANN pasados")
# println("==============================")
# println()

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------


# Establecemos la semilla para que los resultados sean siempre los mismos
using Random: seed!
# Comprobamos que la generación de números aleatorios es la esperada:
seed!(1); @assert(isapprox(rand(), 0.07336635446929285))
#  Si fallase aquí, seguramente dara error al comprobar los resultados de la ejecución de la siguiente función porque depende de la generación de números aleatorios


seed!(1); ann = trainHopfield(rand([-1,1],4,4));
@assert(eltype(ann) == Float32);
@assert(ann == Float32[0.0  0.5  0.0  1.0; 0.5  0.0  0.5  0.5; 0.0  0.5  0.0  0.0; 1.0  0.5  0.0  0.0]);

seed!(1); ann = trainHopfield(randn(4,4).>=0);
@assert(eltype(ann) == Float32);
@assert(ann == Float32[0.0  0.0  -1.0  0.0; 0.0  0.0  0.0  0.0; -1.0  0.0  0.0  0.0; 0.0  0.0  0.0  0.0]);

imagesNCHW = loadImagesNCHW(imageFolder; datasetType=Float64, resolution=2)
ann = trainHopfield(imagesNCHW.>=0.5);
@assert(eltype(ann) == Float32);
@assert(ann == Float32[0.0  -0.5   0.0  -0.5; -0.5   0.0  -0.5   0.0;  0.0  -0.5   0.0   0.5; -0.5   0.0   0.5   0.0]);

println("==============================")
println("Tests trainHopfield pasados")
println("==============================")
println()

seed!(1); S = stepHopfield(ann, rand([-1,1],4))
@assert(eltype(S) == Float32);
@assert(S == Float32[0, 0, 1, 1]);

seed!(1); S = stepHopfield(ann, randn(4).>=0)
@assert(eltype(S) == Bool);
@assert(S == Bool[0, 1, 1, 1]);

println("==============================")
println("Tests stepHopfield pasados")
println("==============================")
println()

imagesNCHW = loadImagesNCHW(imageFolder; datasetType=Float64, resolution=8).>=0.5;

seed!(1); newImagesNCHW = addNoise(imagesNCHW, 0.5);
@assert(eltype(newImagesNCHW)==eltype(imagesNCHW))
@assert(size(newImagesNCHW)==size(imagesNCHW))
@assert(0.45 <= mean(newImagesNCHW .== imagesNCHW) <= 0.55)

println("==============================")
println("Tests addNoise pasados")
println("==============================")
println()

newImagesNCHW = cropImages(imagesNCHW, 0.25);
@assert(eltype(newImagesNCHW)==eltype(imagesNCHW))
@assert(size(newImagesNCHW)==size(imagesNCHW))
@assert(!any(newImagesNCHW[:,:,:,7:8]))

println("==============================")
println("Tests cropImages pasados")
println("==============================")
println()

seed!(1); newImagesNCHW = randomImages(10, 32);
@assert(size(newImagesNCHW)==(10,1,32,32))
@assert(eltype(newImagesNCHW)==Bool)
@assert(0.45 <= mean(newImagesNCHW) <= 0.55)

println("==============================")
println("Tests randomImages pasados")
println("==============================")
println()


MNISTDataset = loadMNISTDataset(datasetFolder; labels=[2,8], datasetType=Float32);

templateInputs, templateLabels = averageMNISTImages(MNISTDataset[1], MNISTDataset[2]);
println(size(templateInputs[1]))
@assert(size(templateInputs)==(2,1,28,28))
@assert(eltype(templateInputs)==Float32)
@assert(length(templateLabels)==2)
@assert(unique(templateLabels)==[2,8])
@assert(eltype(templateLabels)<:Integer)

println("==============================")
println("Tests averageMNISTImages pasados")
println("==============================")
println()

outputLabels = classifyMNISTImages(templateInputs[[2,1],:,:,:].>=0.5, templateInputs.>=0.5, templateLabels);
@assert(eltype(outputLabels)<:Integer)
@assert(       outputLabels==[8,2])

println("==============================")
println("Tests classifyMNISTImages pasados")
println("==============================")
println()

trainingAccuracy, testAccuracy = calculateMNISTAccuracies(datasetFolder, [8,2], 0.5);
@assert(isapprox(trainingAccuracy, 0.8861038191210094));
@assert(isapprox(testAccuracy,     0.8773678963110668));

println("==============================")
println("Tests calculateMNISTAccuracies pasados")
println("==============================")
println()