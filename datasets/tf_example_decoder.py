import tensorflow as tf


class TfExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            'image/encoded/256':
                tf.io.FixedLenFeature((), tf.string),
            'image/encoded/512':
                tf.io.FixedLenFeature((), tf.string),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string),
            'label':
                tf.io.FixedLenFeature((), tf.int64),
        }

    def _decode_image(self, content, channels):
      return tf.image.decode_jpeg(content, channels)

    def decode(self, serialized_example, image_size=256):
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features)
        image = self._decode_image(parsed_tensors[f'image/encoded/{image_size}'], 3)
        label = parsed_tensors['label']
        return image, label
