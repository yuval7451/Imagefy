# Author: Yuval Kaneti

## Imports
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs

class KmeansConfig():
    def __init__(self, base_model_dir):
        self.base_model_dir = base_model_dir

    def _gpu_options(self):
        gpu_options = tf.compat.v1.GPUOptions(
            allow_growth=True,
            force_gpu_compatible=True,
            per_process_gpu_memory_fraction=0.9,
        )
        return gpu_options
 
    def _config_proto(self):        
        config_proto = tf.compat.v1.ConfigProto(
            gpu_options=self._gpu_options(),
            allow_soft_placement=True,
            log_device_placement=False,
        )
        return config_proto
    
    def get_run_config(self):
        config = tf.estimator.RunConfig(
            model_dir=self.base_model_dir,
            save_summary_steps=5,
            keep_checkpoint_max=1,
            #? train_distribute=tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0'),
            session_config=self._config_proto()
        )
        return config
    
def InceptionConfig():
    return tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                force_gpu_compatible=True,
                per_process_gpu_memory_fraction=0.9),
            allow_soft_placement=True,
            log_device_placement=False
        )
