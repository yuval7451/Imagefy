# Imports
# %tensorflow_version 2.x
import logging
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])


# Constants
DATA_DIR = 'D:\\Imagefy\\dataset\\tiny\\*.JPG'
# MODEL_DIR = '/content/gdrive/My Drive/Colab Notebooks/logs/Imagefy_test_modelc'

gpu_options = tf.compat.v1.GPUOptions(
    allow_growth=True,
    force_gpu_compatible=True,
    per_process_gpu_memory_fraction=0.99,
)

config_proto = tf.compat.v1.ConfigProto(
    gpu_options=gpu_options,
    allow_soft_placement=False,
    log_device_placement=False
)

config = tf.estimator.RunConfig(
    save_summary_steps=10,
    keep_checkpoint_max=1,
    # model_dir=self.base_model_dir,
    # train_distribute=tf.contrib.distribute.OneDeviceStrategy(device='/device:GPU:0'),
    session_config=config_proto
)



@tf.function
def _load_tensor(image_path: str):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    image = image - tf.math.reduce_mean(input_tensor=image, axis=0)
    image = image / tf.math.reduce_max(input_tensor=tf.abs(image), axis=0)
    image = tf.reshape(image, [-1])
    return image

@tf.function
def input_fn(dir_path, num_epochs, batch_size=None):
    options = tf.data.Options()
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = True
    dataset = tf.data.Dataset.list_files(dir_path, shuffle=False)
    dataset = dataset.map(_load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).cache().repeat(num_epochs)
    # dataset = dataset.apply((tf.data.experimental.prefetch_to_device('/device:GPU:0', buffer_size=tf.data.experimental.AUTOTUNE))).cache()
    dataset = dataset.with_options(options)
    return dataset  

@tf.function
def run():
    cluster = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=10,
        use_mini_batch=True,
        config=config,
    )
    cluster.train(lambda: input_fn(DATA_DIR, 1, 10))
    cluster_centers = cluster.cluster_centers()
    score = cluster.score(lambda: input_fn(DATA_DIR, 1, 10)) 
    cluster_indices = list(cluster.predict_cluster_index(lambda: input_fn(DATA_DIR, 1, 10)))
    labels = []
    for cluster_index in cluster_indices:
        labels.append(cluster_index)

    # tf.print(labels)
    tf.print(labels)
   


if __name__ == '__main__':
    run()