
import os
import logging
import tensorflow as tf
from integration.utils.data_utils import BaseLoader, TensorLoader
from tensorflow.contrib.tensorboard.plugins import projector


class Tensorboard(BaseLoader):
    def __init__(self, save_dir, metadata_path, **kwargs):
        """
        self.dir_path = dir_path
        self.image_size = image_size
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.dataset = None
        self._loader = None
        self.save_dir = save_dir
        self.metadata_path = metadata_path

        self._get_data()
    def _get_data(self):
        self._loader = TensorLoader(**self.kwargs)
        self.dataset = self._loader.run(batch_size=None, shuffle=False, num_epochs=1)

    def prepare_data(self):
        #tf.compat.v1.data.make_one_shot_iterator(dataset)
        iterator = self.dataset.make_one_shot_iterator()
        embedings = []
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
        # Loop until all elements have been consumed.
            try:
                while True:
                    output = sess.run(next_element)
                    embedings.append(tf.Variable(output, dtype=tf.float32))     
            except tf.errors.OutOfRangeError:
                pass

            embedings = tf.stack(embedings)
            embedding_var = tf.Variable(embedings, name='embedings', trainable=False)
            # return embedding_var2
            writer = tf.compat.v1.summary.FileWriter(self.save_dir, graph=None)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = self.metadata_path
            logging.debug(f"Saving Projector Embeddings")
            projector.visualize_embeddings(writer, config)

            saver = tf.train.Saver(max_to_keep=1)
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, os.path.join(self.save_dir, 'ckpt'))

    def run(self):
        pass

def main():
    dir_path = "D:\\Imagefy\\noa train\\*"
    image_size = 124
    save_dir = "C:\\yuval\\computer\\Projects\\Imagefy\\logs\\MiniBatchKmeansTensorflowWraper-1601806183.8390117"
    metadata_path = "C:\\yuval\\computer\\Projects\\Imagefy\\logs\\MiniBatchKmeansTensorflowWraper-1601806183.8390117\\MiniBatchKmeansTensorflowWraper-metadata.tsv"

    t = Tensorboard(save_dir=save_dir, metadata_path=metadata_path, dir_path=dir_path, image_size=image_size)
    t.prepare_data()
    # def run(self):
        # tensorboard_wraper = TensorboardWraper()

if __name__ == '__main__':
    main()
