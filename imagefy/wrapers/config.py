

import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
            allow_soft_placement=False,
            log_device_placement=False,
        )
        return config_proto
    
    def get_run_config(self):
        config = tf.estimator.RunConfig(
            model_dir=self.base_model_dir,
            save_summary_steps=5,
            keep_checkpoint_max=1,
            # train_distribute=tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0'),
            session_config=self._config_proto()
        )
        return config
    
    def get_hooks(self):
        return [self._summary_hook()] #, self._profile_hook()] tf_debug.TensorBoardDebugHook("localhost:1337") BeholderHook(self.base_model_dir)

    def _summary_hook(self):
        summary_hook = tf.estimator.SummarySaverHook(
            save_steps=1,
            output_dir=os.path.join(self.base_model_dir), #,'summary',
            # summary_op=tf.summary.merge_all()
            scaffold=tf.compat.v1.train.Scaffold(summary_op=tf.compat.v1.summary.merge_all())
        ) 
        return summary_hook
  
    def _profile_hook(self):
        profile_hook = tf.estimator.ProfilerHook(
            # save_secs=10,
            save_steps=10,
            output_dir=os.path.join(self.base_model_dir,'summary'),
            show_dataflow=True,
            show_memory=True
        ) 
        return profile_hook 

    def _metadata_hook(self):
        metadata_hook = MetadataHook(save_steps=1, output_dir=os.path.join(self.base_model_dir)) # 'metadata'
        return metadata_hook

class MetadataHook(SessionRunHook):
    def __init__ (self,
                  save_steps=None,
                  save_secs=None,
                  output_dir=""):
        self._output_tag = "step-{}"
        self._output_dir = output_dir
        self._timer = SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()
        self._writer = tf.compat.v1.summary.FileWriter(self._output_dir, tf.compat.v1.get_default_graph())

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            if self._request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._writer.add_run_metadata(
                run_values.run_metadata, self._output_tag.format(global_step))
            self._writer.flush()
        self._next_step = global_step + 1

    def end(self, session):
        self._writer.close()

def InceptionConfig():
    return tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                force_gpu_compatible=True,
                per_process_gpu_memory_fraction=0.9),
            allow_soft_placement=True,
            log_device_placement=False
        )
        