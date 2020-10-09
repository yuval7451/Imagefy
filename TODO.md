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
- [X] Fix import in data_utils
- [X] Upgrade to TF 2.x
- [X] Add Init or BaseSuit init for GPU's Verbosity logging, etc..
- [X] Profiling, Debug & TF 2.3
- [X] Better Readmes for all branches, fix typo.
- [X] better Documention in all branches
- [X] Automated patamters Tunning
- [X] look into the params of mini batch kmeans
- [X] Dataset prefetching dataset
    **dataset.apply(tf.data.experimental.copy_to_device("/gpu:0")).prefetch()**
    **tf.data.experimental.prefetch_to_device()**
- [] Warm start with kmeans - Kmeans doesnt have warm start...
    *https://stackoverflow.com/questions/49846207/tensorflow-estimator-warm-start-from-and-model-dir*
- [] Revert to tf1.5 and try 
    *https://github.com/GoogleCloudPlatform/tf-estimator-tutorials/blob/master/03_Clustering/*
    *https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb*
    *https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/*
- [] Imagenet autoencoder *https://github.com/anikita/ImageNet_Pretrained_Autoencoder*
- [] Web
    *https://github.com/imadelh/ML-web-app*
    *https://heartbeat.fritz.ai/deploy-a-machine-learning-model-as-a-web-application-part-1-a1c1ff624f7a*
    *https://heartbeat.fritz.ai/deploy-a-machine-learning-model-as-a-web-application-part-2-2f590342b390*
    *https://www.streamlit.io/*
    *https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7*
    *https://medium.com/@alyafey22/train-test-and-deploy-ml-models-in-the-browser-using-one-notebook-ef0e8a7c29e4*
    
- [] Fix GPU issue with inference
- [] tf.distribute.MirroredStrategy
- [] Calculate Number of Clusters on you own len(data) // 10 : //13 ?
- [] WSL2, Cuda, RAPIDS & CudNN
