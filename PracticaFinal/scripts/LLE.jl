
using Pkg
Pkg.activate("../environment")
Pkg.instantiate()

using MLJModelInterface
import ManifoldLearning

mutable struct LLE <: MLJModelInterface.Unsupervised
    k::Int
    d::Int
end

LLE(; k=15, d=2) = LLE(k, d)


function MLJModelInterface.fit(model::LLE, verbosity, X)

    return nothing, nothing, nothing
end


function MLJModelInterface.transform(model::LLE, fitresult, Xnew)
   
    X_mat = MLJModelInterface.matrix(Xnew)'
    

    fitresult = ManifoldLearning.fit(ManifoldLearning.LLE, X_mat; 
                                     k=model.k, 
                                     maxoutdim=model.d)
    
    proyeccion_mat = ManifoldLearning.predict(fitresult)
   
    nombres = Tuple(Symbol("x$i") for i in 1:model.d)
    columnas = tuple([proyeccion_mat[i, :] for i in 1:model.d]...)
    
    return NamedTuple{nombres}(columnas)
end