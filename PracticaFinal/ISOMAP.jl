using MLJModelInterface
using MultivariateStats
using ManifoldLearning

mutable struct Isomap <: MLJModelInterface.Unsupervised
    k::Int
    d::Int
end

Isomap(; k=15, d=2) = Isomap(k, d)


function MLJModelInterface.fit(model::Isomap, verbosity, X)
    return nothing, nothing, nothing
end


function MLJModelInterface.transform(model::Isomap, fitresult, Xnew)

    X_mat = MLJModelInterface.matrix(Xnew)' 
    

    fitresult = ManifoldLearning.fit(ManifoldLearning.Isomap, X_mat; 
                                     k=model.k, 
                                     maxoutdim=model.d)
    
    
    proyectado = ManifoldLearning.predict(fitresult) 
    
    nombres = Tuple(Symbol("x$i") for i in 1:model.d)
    tabla = NamedTuple{nombres}(tuple([proyectado[i, :] for i in 1:model.d]...))
    
    return tabla
end