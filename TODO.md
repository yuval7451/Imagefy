# TODO
- [X] Create a better data loader(Threaded, Queues, Chanked, MemoryAlloc. *...) 
- [X] Think of a way to Store a Table in memory to regroup images
- [X] Fix the Arg parser
- [X] Look into visualiztion and PCA & TSNE on GPU
    *PCA* -> [https://github.com/neheller/TensorFlow-PCA/blob/master/pca.py]
    *TSNE* -> [https://github.com/maestrojeong/t-SNE/blob/master/t-SNE.ipynb]
- [X] Look into other Clustering Algorithems with GPU implemntion
    *Hierarchical Clustering* -> [None]
    *DBSCAN* -> [research\clustering\TF_dbscan.py]
    *Mean-Shift* -> [https://github.com/lephong/tf-meanshift/blob/master/src/meanshift.py]
    *Spectral Clustering* -> [None]
- [X] Look at ProcessPoolExecutor with lower workers for data loading 
    *Threads Win*
- [X] Numda vs numpy (for DataLoader, PCA & ...)
- [X] Change BaseSocre To WraperOutput
- [X] Batched Dataloading
    *tf.data.Dataset.from_tensor_slices ?*
    *Some kind of DataGenerator*
- [X] Scrape AsincIO DataLoader ->
- [X] MiniBatch Kmeans [https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/experimental/KMeans]
- [X] Colab [https://colab.research.google.com/github/rpi-techfundamentals/spring2019-materials/blob/master/01-overview/01-notebook-basics/03-running-code.ipynb]
- [X] Make everything more General
- [X] Argparsing
- [X] Modules, Suits & Wrapers
- [X] tf.data.Dataset
- [X] Move tensorboard to Wrapers?
- [X] Hollow Image..
- [X] Fix tensorboard logs
- [X] Improve README
- [X] Tensorboard for tensor_loader
- [X] Hooks
- [X] Config
- [X] tf.funcion
- [X] add Documention
- [X] Big Data test
- [X] Look at converting keras model to estimator
- [] Fix import in data_utils
- [] GPU issue with inference
- [] Upgrade to TF 2.x
- [] Better readmes for all branches
- [] Calculate Number of Clusters on you own
- [] WSL2, Cuda, RAPIDS & CudNN
- [] Profiling, Debug & TF 2.3

# Sources
- [Source][https://www.baeldung.com/cs/clustering-unknown-number']