using MLJModelInterface
using TSne

# 1. Define the Model Struct with hyperparameters
mutable struct TSNE <: MLJModelInterface.Unsupervised
    d::Int
    perplexity::Float64
    max_iter::Int
    pca_init::Bool
end

# 2. Set default values
TSNE(; d=2, perplexity=30.0, max_iter=1000, pca_init=true) = 
    TSNE(d, perplexity, max_iter, pca_init)

# 3. Define the fit logic
function MLJModelInterface.fit(model::TSNE, verbosity, X)

    return nothing, nothing, nothing
end

function MLJModelInterface.transform(model::TSNE, fitresult, Xnew)
    X_matrix = MLJModelInterface.matrix(Xnew)
    
    embedding = TSne.tsne(
        Float64.(X_matrix), 
        model.d, 
        0, 
        model.max_iter, 
        model.perplexity;
        pca_init = model.pca_init,
        verbose = false,
        progress = false
    )

    return embedding
end


MLJModelInterface.metadata_model(TSNE,
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    output=MLJModelInterface.Table(MLJModelInterface.Continuous),
    descr="t-Distributed Stochastic Neighbor Embedding."
)