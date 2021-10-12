import tensorflow as tf


def random_brightness(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_brightness(image, max_delta=0.1, seed=seed), label

def random_contrast(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_contrast(image, lower=0.8, upper=2, seed=seed), label

def random_hue(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_hue(image, max_delta=0.2, seed=seed), label

def random_saturation(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_saturation(image, lower=0.2, upper=2.5, seed=seed), label


def make_random_transformation(image, label):
    rand = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )

    borders = tf.constant([0.25, 0.5, 0.75, 1])

    hue_fn = lambda: random_hue(image, label)
    saturation_fn = lambda: random_saturation(image, label)
    brightness_fn = lambda: random_brightness(image, label)
    contrast_fn = lambda: random_contrast(image, label)

    image, label = tf.case([(tf.less(rand, borders[0]), hue_fn),
                            (tf.less(rand, borders[1]), saturation_fn),
                            (tf.less(rand, borders[2]), brightness_fn),
                            (tf.less(rand, borders[3]), contrast_fn)], exclusive=False)
    return image, label


def create_input(image, label):
    image = tf.keras.applications.resnet.preprocess_input(image)
    label = tf.cast(label, tf.int32)
    return image, label