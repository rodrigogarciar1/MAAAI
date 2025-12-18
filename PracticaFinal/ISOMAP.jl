using MLJModelInterface
using MultivariateStats
using ManifoldLearning # Asegúrate de tenerlo cargado

mutable struct Isomap <: MLJModelInterface.Unsupervised
    k::Int
    d::Int
end

Isomap(; k=15, d=2) = Isomap(k, d)

# 1. El entrenamiento ocurre AQUÍ
function MLJModelInterface.fit(model::Isomap, verbosity, X)
    return nothing, nothing, nothing
end

# 2. El transform usa el fitresult y devuelve una NamedTuple
function MLJModelInterface.transform(model::Isomap, fitresult, Xnew)
    # fitresult es el objeto Isomap ya entrenado
    # predict() o simplemente acceder a los componentes proyectados
    # En ManifoldLearning, fit() ya devuelve las coordenadas reducidas

    X_mat = MLJModelInterface.matrix(Xnew)' # Transponer para MultivariateStats/ManifoldLearning (p x n)
    
    # Entrenamos el modelo real
    fitresult = ManifoldLearning.fit(ManifoldLearning.Isomap, X_mat; 
                                     k=model.k, 
                                     maxoutdim=model.d)
    
    
    proyectado = ManifoldLearning.predict(fitresult) # Obtenemos la matriz (d x n)
    
    # Convertimos a NamedTuple (Formato de tabla: columnas como vectores)
    # Esto crea nombres automáticos x1, x2, ..., xd
    nombres = Tuple(Symbol("x$i") for i in 1:model.d)
    tabla = NamedTuple{nombres}(tuple([proyectado[i, :] for i in 1:model.d]...))
    
    return tabla
end