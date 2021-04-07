
import tensorflow as tf

class BiasLayer(tf.keras.layers.Layer):
    def __init__(
        self, bias_init=0.0, name=None, 
        trainable=True, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.bias = tf.Variable(
            initial_value=bias_init, 
            trainable=trainable, name=name)
    
    def call(self, x):
        return x + self.bias
