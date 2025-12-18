using MLJModelInterface
import ManifoldLearning # Usualmente LLE reside aquí

mutable struct LLE <: MLJModelInterface.Unsupervised
    k::Int
    d::Int
end

LLE(; k=15, d=2) = LLE(k, d)

# 1. El aprendizaje del manifold ocurre en fit
function MLJModelInterface.fit(model::LLE, verbosity, X)

    return nothing, nothing, nothing
end

# 2. Transform devuelve la proyección en formato NamedTuple (Tabla)
function MLJModelInterface.transform(model::LLE, fitresult, Xnew)
    # En algoritmos de Manifold Learning clásicos, la proyección 
    # ya está calculada en el fitresult. 
    # Usamos predict() para extraer las coordenadas reducidas (matriz d x n)

    # MLJ entrega X como tabla, lo convertimos a matriz p x n (transpuesta)
    X_mat = MLJModelInterface.matrix(Xnew)'
    
    # Entrenamos el modelo LLE
    # fitresult guardará la estructura del manifold calculado
    fitresult = ManifoldLearning.fit(ManifoldLearning.LLE, X_mat; 
                                     k=model.k, 
                                     maxoutdim=model.d)
    
    proyeccion_mat = ManifoldLearning.predict(fitresult)
    
    # Convertimos la matriz de salida a una NamedTuple de vectores
    # Esto genera columnas :x1, :x2, ..., :xd
    nombres = Tuple(Symbol("x$i") for i in 1:model.d)
    columnas = tuple([proyeccion_mat[i, :] for i in 1:model.d]...)
    
    return NamedTuple{nombres}(columnas)
end