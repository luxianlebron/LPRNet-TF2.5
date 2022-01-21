import tensorflow as tf
import tensorflow.keras.layers as ly


def small_basic_block(filters, channels=32, name=None):
    x = inputs = tf.keras.Input(shape=(None, None, channels))
    x = ly.Conv2D(filters=filters // 2, kernel_size=3, strides=1, padding='same')(x)
    x = ly.BatchNormalization()(x)
    x = ly.ReLU()(x)
    x = ly.DepthwiseConv2D(kernel_size=1, strides=1, padding='same')(x)
    x = ly.BatchNormalization()(x)
    x = ly.ReLU()(x)
    x = ly.DepthwiseConv2D(kernel_size=1, strides=1, padding='same')(x)
    x = ly.ReLU()(x)
    x = ly.Concatenate(axis=-1)([x, inputs])
    x = ly.Conv2D(filters=filters, kernel_size=3, strides=1)(x)
    x = ly.BatchNormalization()(x)
    outputs = ly.ReLU()(x)
    return tf.keras.Model(inputs, outputs)


def LPRNet(num_chars, dropout_rate=0.5):
    block1 = small_basic_block(64, 32)
    block2 = small_basic_block(64, 64)
    x = inputs = tf.keras.Input([24, 94, 3], name='input')
    x = ly.Conv2D(filters=32, kernel_size=3, strides=1)(x)
    x = ly.BatchNormalization()(x)
    x = ly.ReLU()(x)
    x = block1(x)
    x = block2(x)
    x = ly.Conv2D(filters=64, kernel_size=3, strides=(1, 2))(x)
    x = ly.Dropout(dropout_rate)(x)
    x = ly.BatchNormalization()(x)
    x = ly.ReLU()(x)
    x = ly.AveragePooling2D(pool_size=3, strides=2)(x)
    x = ly.Conv2D(filters=num_chars, kernel_size=1, strides=1, name='container')(x)
    x = tf.transpose(x, [0, 3, 1, 2])
    outputs = tf.reduce_mean(x, axis=2, name='output')
    return tf.keras.Model(inputs, outputs)
