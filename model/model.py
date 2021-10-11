import tensorflow as tf
import numpy as np

class RotNet90:
    def __init__(self, size):
        input_layer = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        core = tf.keras.applications.ResNet50(
            include_top=True, weights=None, input_tensor=input_layer,
            input_shape=None, pooling=None, classes=4, classifier_activation='softmax'
        )
        x = core(x)
        self.model = tf.keras.models.Model(inputs=[input_layer], outputs=[x])
        self.size = size

class ResnetModel:
    def __init__(self, checkpoint_path: str, size: int=256):
        input_layer = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        x = tf.cast(input_layer, tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        core = tf.keras.applications.ResNet50(
            include_top=True, weights=None, input_tensor=x,
            input_shape=None, pooling=None, classes=4, classifier_activation='softmax'
        )
        x = core(x)
        self.model = tf.keras.models.Model(inputs=[input_layer], outputs=[x])
        self.model.compile()
        self.model.load_weights(checkpoint_path)
        self.size = size

    def predict_for_image(self, image: np.ndarray) -> np.ndarray:
        image_tensor = tf.image.resize(image, size=tf.constant([self.size, self.size]), method=tf.image.ResizeMethod.LANCZOS3)
        predictions = self.model.predict(tf.expand_dims(image_tensor, axis=0))
        predictions = tf.squeeze(predictions)
        return tf.math.argmax(predictions).numpy()
        # return predictions


class ResnetModelWithAug:
    def __init__(self, checkpoint_path: str, size: int=256):
        input_layer = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        x = tf.cast(input_layer, tf.float32)
        random_translation = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1)
        random_zoom = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
        random_contrast = tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)
        x = random_translation(x)
        x = random_zoom(x)
        x = random_contrast(x)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        core = tf.keras.applications.ResNet50(
            include_top=True, weights=None, input_tensor=x,
            input_shape=None, pooling=None, classes=4, classifier_activation='softmax'
        )
        x = core(x)
        self.model = tf.keras.models.Model(inputs=[input_layer], outputs=[x])
        self.model.compile()
        self.model.load_weights(checkpoint_path)
        self.size = size

    def predict_for_image(self, image: np.ndarray) -> np.ndarray:
        image_tensor = tf.image.resize(image, size=tf.constant([self.size, self.size]), method=tf.image.ResizeMethod.LANCZOS3)
        predictions = self.model(tf.expand_dims(image_tensor, axis=0), training=None)
        predictions = tf.squeeze(predictions)
        return tf.math.argmax(predictions).numpy()
        # return predictions


class ResnetFromSavedModel:
    def __init__(self, saved_model_path: str, size: int = 256):
        self.model = tf.saved_model.load(saved_model_path)
        self.size = size

    def predict_for_image(self, image: np.ndarray) -> np.ndarray:
        # image_tensor = tf.image.resize(image, size=tf.constant([self.size, self.size]),
        #                                method=tf.image.ResizeMethod.LANCZOS3)
        # image_tensor = tf.cast(image, tf.uint8)
        predictions = self.model.signatures["serving_default"](input_1=tf.expand_dims(image, axis=0))['resnet50']
        predictions = tf.squeeze(predictions)
        return tf.math.argmax(predictions).numpy()

