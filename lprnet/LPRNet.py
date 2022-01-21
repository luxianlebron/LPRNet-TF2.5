import tensorflow as tf
import tensorflow.keras.layers as ly


# 定义基本模块
class small_basic_block(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(small_basic_block, self).__init__()
        self.block = tf.keras.Sequential([
            ly.Conv2D(filters=filters // 4, kernel_size=1, strides=1),
            ly.BatchNormalization(),
            ly.ReLU(),
            ly.Conv2D(filters=filters // 4, kernel_size=(3, 1), strides=1, padding='same'),
            ly.BatchNormalization(),
            ly.ReLU(),
            ly.Conv2D(filters=filters // 4, kernel_size=(1, 3), strides=1, padding='same'),
            ly.BatchNormalization(),
            ly.ReLU(),
            ly.Conv2D(filters=filters, kernel_size=1, strides=1)
        ])
    def call(self, inputs):
        return self.block(inputs)


# lpr网络构建
class LPRNet(tf.keras.Model):
    def __init__(self, lpr_len, class_num, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.lpr_len = lpr_len
        self.class_num = class_num
        self.dropout_rate = dropout_rate

        # 第1个 inception
        self.icp_1 = tf.keras.Sequential([
            ly.Conv2D(filters=64, kernel_size=3, strides=1, name='input'),
            ly.BatchNormalization(),
            ly.ReLU()
        ]) 
        # 第2个 inception
        self.icp_2 = tf.keras.Sequential([
            ly.MaxPool2D(pool_size=3, strides=1),
            small_basic_block(filters=128),
            ly.BatchNormalization(),
            ly.ReLU()
        ])
        # 第3个 inception
        self.icp_3 = tf.keras.Sequential([
            ly.MaxPool2D(pool_size=(3, 3), strides=(1, 2)),
            small_basic_block(filters=256),
            ly.BatchNormalization(),
            ly.ReLU(),
            small_basic_block(filters=256),
            ly.BatchNormalization(),
            ly.ReLU()
        ])
        # 第4个 inception
        self.icp_4 = tf.keras.Sequential([
            ly.MaxPool2D(pool_size=(3, 3), strides=(1, 2)),
            ly.Dropout(rate=dropout_rate),
            ly.Conv2D(filters=256, kernel_size=(1, 4), strides=1),
            ly.BatchNormalization(),
            ly.ReLU(),
            ly.Dropout(rate=dropout_rate),
            ly.Conv2D(filters=class_num, kernel_size=(13, 1), strides=1),
            ly.BatchNormalization(),
            ly.ReLU()
        ])

        self.container = tf.keras.Sequential([
            ly.Conv2D(filters=class_num, kernel_size=1, strides=1),
#             ly.BatchNormalization(),
#             ly.ReLU(),
            # ly.Conv2D(filters=lpr_len + 1, kernel_size=3, strides=2),
            # ly.ReLU()
        ])

    def call(self, inputs):
        icp_1 = self.icp_1(inputs)
        icp_2 = self.icp_2(icp_1)
        icp_3 = self.icp_3(icp_2)
        icp_4 = self.icp_4(icp_3)

        global_context = list()
        for i, f in enumerate([icp_1, icp_2, icp_3, icp_4]):
            if i in [0, 1]:
                f = ly.AveragePooling2D(pool_size=5, strides=5)(f)
            if i == 2:
                f = ly.AveragePooling2D(pool_size=(4, 10), strides=(4, 2))(f)
            f_pow = tf.multiply(f, f)
            f_mean = tf.reduce_mean(f_pow)
            f = tf.divide(f, f_mean)
            global_context.append(f)

        x = tf.concat(global_context, axis=3)
        x = self.container(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        output = tf.reduce_mean(x, axis=2)
#         output = tf.keras.layers.Softmax(axis=1)(output)

        return output




