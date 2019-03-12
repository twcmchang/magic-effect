import os
import numpy as np
import tensorflow as tf
from PIL import Image

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

    def __init__(self, model_dir):
        """Creates and loads pretrained deeplab model."""
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def

        self.graph = tf.Graph()
        with tf.gfile.GFile(os.path.join(model_dir, self.FROZEN_GRAPH_NAME), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with self.graph.as_default():
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="")
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = np.array(batch_seg_map[0] == 15, np.uint8) * 255
        seg_map = np.stack((seg_map, seg_map, seg_map),axis=2).astype(np.uint8)
        rgb_img = np.array(resized_image)
        return rgb_img, seg_map

model_dir = '/Users/cmchang/magic-effect/model/deeplabv3_mnv2_pascal_train_aug/'
MODEL = DeepLabModel(model_dir)
print('model loaded successfully!')